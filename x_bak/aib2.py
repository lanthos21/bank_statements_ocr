# aib.py
# AIB parser with numeric ROI re-OCR, row-consistency rescue, and debug prints.

from __future__ import annotations

import re
import hashlib
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
import cv2
import pytesseract
import numpy as np

from utils import parse_currency, parse_date, date_variants

# ----------------------------
# Debug helper
# ----------------------------
def _dbg(enabled: bool, msg: str):
    if enabled:
        print(msg)

# ----------------------------
# Regexes
# ----------------------------

RE_BAL_FWD   = re.compile(r"\bBALANCE\s*FORWARD\b", re.IGNORECASE)
RE_DATE_LONG = re.compile(r"\b(\d{1,2})\s+[A-Za-z]{3,9}\s+\d{4}\b")
RE_DATE_NUM  = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")

CURRENCY_HEADER_TOKENS = {"€", "£", "EUR", "GBP"}

def _is_currency_token(tok: str) -> bool:
    if tok is None:
        return False
    t = tok.strip().upper()
    return t in CURRENCY_HEADER_TOKENS

def _find_date_in_text(s: str) -> Optional[str]:
    if not s:
        return None
    m = RE_DATE_LONG.search(s)
    if m:
        return parse_date(m.group(0))
    m = RE_DATE_NUM.search(s)
    if m:
        d, mth, y = m.groups()
        if len(y) == 2:
            y = "20" + y
        try:
            return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
        except Exception:
            return None
    return None

# ----------------------------
# Header / column detection
# ----------------------------

def detect_column_positions(lines: list[dict]) -> tuple[dict, int, int] | None:
    def right_edge(w: dict) -> int:
        return int(w.get("left", 0)) + int(w.get("width", 0))

    for idx, line in enumerate(lines or []):
        words = line.get("words", []) or []
        if not words:
            continue

        pos = {
            "date_left": None,
            "details_left": None,
            "debit_right": None,
            "credit_right": None,
            "balance_right": None,
        }

        n = len(words)
        for i, w in enumerate(words):
            t = (w.get("text") or "").strip().lower()
            if not t:
                continue

            if t == "date" and pos["date_left"] is None:
                pos["date_left"] = int(w.get("left", 0))

            elif t == "details" and pos["details_left"] is None:
                pos["details_left"] = int(w.get("left", 0))

            elif t == "debit" and pos["debit_right"] is None:
                pos["debit_right"] = (
                    right_edge(words[i + 1])
                    if i + 1 < n and _is_currency_token(words[i + 1].get("text"))
                    else right_edge(w)
                )

            elif t == "credit" and pos["credit_right"] is None:
                pos["credit_right"] = (
                    right_edge(words[i + 1])
                    if i + 1 < n and _is_currency_token(words[i + 1].get("text"))
                    else right_edge(w)
                )

            elif t == "balance" and pos["balance_right"] is None:
                pos["balance_right"] = (
                    right_edge(words[i + 1])
                    if i + 1 < n and _is_currency_token(words[i + 1].get("text"))
                    else right_edge(w)
                )

        if all(pos[k] is not None for k in ("date_left", "details_left", "debit_right", "credit_right", "balance_right")):
            return pos, idx, len(lines)

    return None

def categorise_amount_by_right_edge(x_right: int, header_positions: dict, margin: int = 140) -> str:
    x_right = int(x_right)
    best = ("unknown", margin + 1)
    for k, label in (("debit_right", "debit"), ("credit_right", "credit"), ("balance_right", "balance")):
        pos = header_positions.get(k)
        if pos is None:
            continue
        d = abs(x_right - int(pos))
        if d < best[1]:
            best = (label, d)
    return best[0] if best[1] <= margin else "unknown"

# ----------------------------
# Helpers
# ----------------------------

def _right_edge(w: dict) -> int:
    return int(w.get("left", 0)) + int(w.get("width", 0))

def _rightmost_numeric_with_x(words: list[dict]) -> tuple[Optional[int], Optional[float]]:
    best = None
    tokens = words or []
    n = len(tokens)

    for i, w in enumerate(tokens):
        t = (w.get("text") or "").strip()
        if not t:
            continue
        v = parse_currency(t, strip_currency=False)
        xr = _right_edge(w)
        if v is None and i + 1 < n and _is_currency_token((tokens[i + 1].get("text") or "").strip()):
            v = parse_currency(t + (tokens[i + 1].get("text") or ""), strip_currency=False)
            xr = _right_edge(tokens[i + 1])
        if v is None:
            continue
        if best is None or xr > best[0]:
            best = (xr, float(v))
    return (None, None) if best is None else best

AMOUNT_TOKEN_RE = re.compile(
    r"""
    ^[+\-]?
    (?:(?:\d{1,3}(?:[.,]\d{3})+|\d+))
    [.,]\d{2}$
    """,
    re.VERBOSE,
)

def _looks_like_amount_token(tok: str) -> bool:
    if not tok:
        return False
    t = tok.strip()
    if not t:
        return False
    if t[0] in {"€", "£"}:
        t = t[1:]
    elif t[-1:] in {"€", "£"}:
        t = t[:-1]
    return bool(AMOUNT_TOKEN_RE.match(t))

# ----------------------------
# OCR second-pass helpers (numeric-only ROI)
# ----------------------------

def _normalize_ws_and_minus(s: str) -> str:
    return (
        (s or "")
        .replace("\u00A0", " ")
        .replace("\u2009", " ")
        .replace("\u202F", " ")
        .replace("−", "-")
    )

def _window_text(words: list[dict], window: tuple[int, int], pad: int = 8) -> str:
    c0, c1 = float(window[0]) - pad, float(window[1]) + pad
    toks = []
    for w in words or []:
        L = int(w.get("left", 0)); W = int(w.get("width", 0))
        cx = L + 0.5 * W
        if c0 <= cx <= c1:
            t = (w.get("text") or "")
            if any(ch.isdigit() for ch in t) or re.search(r"[.,+-/OoIl]", t):
                toks.append(t)
    return "".join(toks)

def _should_run_roi(win_str: str, val_win_is_none: bool) -> bool:
    s = _normalize_ws_and_minus(win_str)
    has_digit = any(ch.isdigit() for ch in s)
    if not has_digit:
        return False
    if val_win_is_none:
        return True
    if any(c in s for c in "/OoIl"):
        return True
    if re.search(r"[.,]\s*/|/$|/\s*\d$", s):
        return True
    return False

def _rightmost_num_run_bounds(
    words: list[dict],
    window: tuple[int, int],
    pad: int = 10,
    merge_gap_px: int = 28
) -> tuple[Optional[int], Optional[int]]:
    c0, c1 = float(window[0]) - pad, float(window[1]) + pad
    cand = []
    for w in words or []:
        txt = str(w.get("text") or "")
        if not any(ch.isdigit() for ch in txt):
            continue
        L = int(w.get("left", 0)); W = int(w.get("width", 0))
        cx = L + 0.5 * W
        if c0 <= cx <= c1:
            cand.append(w)
    if not cand:
        return (None, None)
    cand.sort(key=lambda w: int(w.get("left", 0)))

    runs, cur, last_r = [], [], None
    for w in cand:
        l = int(w.get("left", 0)); r = l + int(w.get("width", 0))
        if last_r is None or (l - last_r) <= merge_gap_px:
            cur.append(w)
        else:
            if cur: runs.append(cur)
            cur = [w]
        last_r = r
    if cur: runs.append(cur)

    run = runs[-1]
    x0 = min(int(w.get("left", 0)) for w in run)
    x1 = max(int(w.get("left", 0)) + int(w.get("width", 0)) for w in run)
    return (x0, x1)

def _numeric_ocr_roi(
    img_path: str,
    x0: int, x1: int, y0: int, y1: int,
    pad: int = 4,
    require_decimal: bool = True,
    require_two_decimals: bool = True,
) -> Optional[float]:
    if not img_path or y0 is None or y1 is None:
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    H, W = img.shape[:2]
    xa = max(0, int(x0) - pad); xb = min(W, int(x1) + pad)
    ya = max(0, int(y0) - pad); yb = min(H, int(y1) + pad)
    if xa >= xb or ya >= yb:
        return None
    roi = img[ya:yb, xa:xb]

    config = (
        "--oem 1 --psm 7 "
        "-c tessedit_char_whitelist=0123456789.,- "
        "-c classify_bln_numeric_mode=1 "
        "-c preserve_interword_spaces=1"
    )
    raw = (pytesseract.image_to_string(roi, config=config) or "").strip()
    raw = raw.replace("−", "-").replace(" ", "")

    if require_decimal and (("." not in raw) and ("," not in raw)):
        return None
    if require_two_decimals and not re.search(r"[.,]\d{2}\b", raw):
        return None

    return parse_currency(raw, strip_currency=False)

def _amount_with_roi_fallback_by_right_edge(
    words: list[dict],
    col_right: int,
    page_img_path: Optional[str],
    line_y0: Optional[int],
    line_y1: Optional[int],
    first_pass_val: Optional[float],
    band: int = 140,
    require_decimal: bool = True,
    force: bool = False,
    debug: bool = False,
    tag: str = "",
) -> Optional[float]:
    window = (int(col_right) - band, int(col_right) + 6)
    win_str = _window_text(words, window, pad=8)
    suspicious = _should_run_roi(win_str, val_win_is_none=(first_pass_val is None))
    need_roi = force or suspicious

    if not need_roi or not page_img_path or line_y0 is None or line_y1 is None:
        return first_pass_val

    rx0, rx1 = _rightmost_num_run_bounds(words, window, pad=10, merge_gap_px=28)
    if rx0 is not None and rx1 is not None:
        x0, x1 = rx0 - 4, rx1 + 4
        _dbg(debug, f"[ROI]{tag} suspicious={suspicious} force={force} crop=tight ({x0},{x1},{line_y0},{line_y1}) raw='{win_str}' fp={first_pass_val}")
    else:
        x1 = window[1] + 6
        x0 = max(window[0] - 6, x1 - band)
        _dbg(debug, f"[ROI]{tag} suspicious={suspicious} force={force} crop=band  ({x0},{x1},{line_y0},{line_y1}) raw='{win_str}' fp={first_pass_val}")

    val_roi = _numeric_ocr_roi(
        page_img_path, int(x0), int(x1), line_y0, line_y1,
        pad=2, require_decimal=require_decimal, require_two_decimals=True
    )

    if val_roi is not None and first_pass_val is not None:
        if abs(float(val_roi) - float(first_pass_val)) <= 0.005:
            _dbg(debug, f"[ROI]{tag} same-as-first-pass -> keep {first_pass_val}")
            return first_pass_val

        if force and not suspicious:
            a, b = abs(float(first_pass_val)), abs(float(val_roi))
            denom = max(a, 1e-6)
            ratio = b / denom
            if ratio < 0.5 or ratio > 2.0:
                _dbg(debug, f"[ROI]{tag} forced but implausible jump ratio={ratio:.3f} -> keep {first_pass_val}")
                return first_pass_val

        _dbg(debug, f"[ROI]{tag} ACCEPT roi={val_roi} from fp={first_pass_val}")
        return float(val_roi)

    if val_roi is not None and first_pass_val is None:
        _dbg(debug, f"[ROI]{tag} ACCEPT roi={val_roi} (no first pass)")
    elif val_roi is None:
        _dbg(debug, f"[ROI]{tag} roi=None -> keep first_pass={first_pass_val}")
    return float(val_roi) if val_roi is not None else first_pass_val

# ----------------------------
# Opening & Closing balances
# ----------------------------

def extract_opening_balance_and_start_date(pages: list[dict], debug: bool = False) -> tuple[float | None, str | None]:
    opening_value: Optional[float] = None
    earliest_date: Optional[str] = None

    for pidx, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        for i, line in enumerate(lines):
            lt = line.get("line_text") or ""
            if not RE_BAL_FWD.search(lt):
                continue

            if opening_value is not None:
                return opening_value, earliest_date

            words = line.get("words", []) or []
            _xr, val = _rightmost_numeric_with_x(words)
            if val is not None:
                opening_value = float(val)
            d = _find_date_in_text(lt)
            if d:
                earliest_date = d
            _dbg(debug, f"[OPENING] seed={opening_value} date={earliest_date}")
            return opening_value, earliest_date

    return None, None

def extract_closing_balance(pages: list[dict]) -> Optional[float]:
    best = None
    headers = {pidx: (detect_column_positions(p.get("lines", []) or []) or (None,))[0] for pidx, p in enumerate(pages)}
    for pidx, page in enumerate(pages or []):
        pos = headers.get(pidx)
        if not pos:
            continue
        bal_r = int(pos["balance_right"])
        margin = 120
        for line in page.get("lines", []) or []:
            xr, val = _rightmost_numeric_with_x(line.get("words", []) or [])
            if val is None or xr is None:
                continue
            if abs(int(xr) - bal_r) <= margin:
                best = float(val)
    return best

def _band_window_for_right_edge(x_right: int, band: int = 140) -> tuple[float, float]:
    """Make a [L,R] window whose right boundary aligns to a column's right edge."""
    xr = int(x_right)
    return (xr - int(band), xr)

def _suspicious_text_by_right_edge(words: list[dict], x_right: int, band: int = 140) -> bool:
    """Heuristic: true if the raw window text hints at OCR trouble (/ Oo Il, trailing slash)."""
    win = _band_window_for_right_edge(x_right, band=band)
    raw = _window_text(words, win, pad=6)
    s = _normalize_ws_and_minus(raw)
    has_digit = any(ch.isdigit() for ch in s)
    if not has_digit:
        return False
    if any(c in s for c in "/OoIl"):
        return True
    if re.search(r"[.,]\s*/|/$|/\s*\d$", s):
        return True
    return False


# ----------------------------
# Transactions (with ROI + row-consistency rescue)
# ----------------------------

def parse_transactions(
    pages: list[dict],
    iban: str | None = None,
    opening_seed: Optional[float] = None,
    debug: bool = False,
) -> tuple[list[dict], Optional[float]]:
    all_transactions: List[dict] = []
    seq = 0
    prev_balance: Optional[float] = opening_seed
    last_tx_balance: Optional[float] = None

    ROW_RESID_TOL = 0.05  # 5c tolerance for a row reconciliation

    _dbg(debug, f"[TX] opening_seed={opening_seed}")

    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_positions(lines)
        if not header:
            continue

        header_positions, start_idx, end_idx = header
        details_left  = int(header_positions["details_left"])
        debit_right   = int(header_positions["debit_right"])
        credit_right  = int(header_positions["credit_right"])
        balance_right = int(header_positions["balance_right"])

        _dbg(debug, f"[HDR] page={page_num+1} pos: debitR={debit_right} creditR={credit_right} balanceR={balance_right}")

        last_seen_date: Optional[str] = None
        page_img_path = page.get("raster_path")

        for line in lines[start_idx + 1: end_idx]:
            ltxt = line.get("line_text", "") or ""
            words = line.get("words", []) or []
            if not words:
                continue

            line_y0 = line.get("y0"); line_y1 = line.get("y1")

            # BALANCE FORWARD rows: seed prev_balance from Balance column (with ROI)
            if RE_BAL_FWD.search(ltxt):
                d = _find_date_in_text(ltxt)
                if d:
                    last_seen_date = d

                dfh = pd.DataFrame(words)
                if not dfh.empty:
                    dfh["right"] = dfh["left"] + dfh["width"]
                    dfh["amount_val"] = dfh["text"].apply(lambda x: parse_currency(x, strip_currency=False))
                    dfh["category"] = dfh.apply(lambda r: categorise_amount_by_right_edge(int(r["right"]), header_positions), axis=1)
                    bal_vals = dfh.loc[(dfh["amount_val"].notna()) & (dfh["category"] == "balance"), ["right","amount_val"]]
                    bal_fp = float(bal_vals.sort_values("right")["amount_val"].iloc[-1]) if not bal_vals.empty else None
                else:
                    bal_fp = None

                prev_before = prev_balance
                prev_balance = _amount_with_roi_fallback_by_right_edge(
                    words, balance_right, page_img_path, line_y0, line_y1,
                    bal_fp, band=140, require_decimal=True, force=True, debug=debug, tag="[BAL-FWD]"
                ) or prev_balance
                _dbg(debug, f"[BAL-FWD] fp={bal_fp} -> prev_balance={prev_balance} (was {prev_before})")
                continue

            # Update date if present
            d = _find_date_in_text(ltxt)
            if d:
                last_seen_date = d
            if last_seen_date is None:
                continue

            df = pd.DataFrame(words)
            if df.empty:
                continue

            df["right"] = df["left"] + df["width"]
            df["amount_val"] = df["text"].apply(lambda x: parse_currency(x, strip_currency=False))
            df["category"] = df.apply(lambda r: categorise_amount_by_right_edge(int(r["right"]), header_positions), axis=1)

            debit_vals   = df.loc[(df["amount_val"].notna()) & (df["category"] == "debit"),   ["right","amount_val"]]
            credit_vals  = df.loc[(df["amount_val"].notna()) & (df["category"] == "credit"),  ["right","amount_val"]]
            balance_vals = df.loc[(df["amount_val"].notna()) & (df["category"] == "balance"), ["right","amount_val"]]

            debit_fp   = float(debit_vals.sort_values("right")["amount_val"].iloc[-1])   if not debit_vals.empty   else None
            credit_fp  = float(credit_vals.sort_values("right")["amount_val"].iloc[-1])  if not credit_vals.empty  else None
            balance_fp = float(balance_vals.sort_values("right")["amount_val"].iloc[-1]) if not balance_vals.empty else None

            # First pass + ROI fallback for each cell
            debit   = _amount_with_roi_fallback_by_right_edge(words, debit_right,   page_img_path, line_y0, line_y1, debit_fp,   band=140, require_decimal=True, debug=debug, tag=f"[row{seq}:debit]")
            credit  = _amount_with_roi_fallback_by_right_edge(words, credit_right,  page_img_path, line_y0, line_y1, credit_fp,  band=140, require_decimal=True, debug=debug, tag=f"[row{seq}:credit]")
            balance = _amount_with_roi_fallback_by_right_edge(words, balance_right, page_img_path, line_y0, line_y1, balance_fp, band=140, require_decimal=True, debug=debug, tag=f"[row{seq}:balance]")

            # Is it a transaction row?
            is_tx = ((debit or 0.0) > 0.0) or ((credit or 0.0) > 0.0)
            if not is_tx:
                if balance is not None:
                    prev_balance = float(balance)
                    # do NOT feed last_tx_balance from non-transaction rows
                continue

            accepted_balance_cell = False  # only True if we accept the cell (first-pass or ROI), not when we override

            # If we can reconcile with prev_balance, check and try rescue once
            if prev_balance is not None and balance is not None:
                expected = round(prev_balance + (credit or 0.0) - (debit or 0.0), 2)
                resid = round(balance - expected, 2)

                if abs(resid) > ROW_RESID_TOL:
                    _dbg(debug, f"[ROW]#{seq} inconsistency: prev={prev_balance} debit={debit} credit={credit} bal={balance} -> expected={expected} resid={resid}")

                    # One forced ROI triple
                    d2 = _amount_with_roi_fallback_by_right_edge(words, debit_right,   page_img_path, line_y0, line_y1, debit,   band=140, require_decimal=True, force=True, debug=debug, tag=f"[row{seq}:debit*]")
                    c2 = _amount_with_roi_fallback_by_right_edge(words, credit_right,  page_img_path, line_y0, line_y1, credit,  band=140, require_decimal=True, force=True, debug=debug, tag=f"[row{seq}:credit*]")
                    b2 = _amount_with_roi_fallback_by_right_edge(words, balance_right, page_img_path, line_y0, line_y1, balance, band=140, require_decimal=True, force=True, debug=debug, tag=f"[row{seq}:balance*]")

                    def score(dv, cv, bv):
                        if bv is None:
                            return float("inf")
                        return abs(round((prev_balance + (cv or 0.0) - (dv or 0.0)) - bv, 2))

                    s1 = score(debit, credit, balance)
                    s2 = score(d2, c2, b2)

                    if s2 + 0.01 < s1:
                        _dbg(debug, f"[ROW]#{seq} ACCEPT roi triple: resid {s1:.2f} -> {s2:.2f}")
                        debit, credit, balance = d2, c2, b2
                        accepted_balance_cell = True
                        # recompute residual to maybe log a keep
                        expected = round(prev_balance + (credit or 0.0) - (debit or 0.0), 2)
                        resid = round((balance or expected) - expected, 2)

                    # If still inconsistent, only override when the text looks suspicious
                    if abs(resid) > ROW_RESID_TOL:
                        suspicious_bal = _suspicious_text_by_right_edge(words, balance_right, band=140)
                        if suspicious_bal:
                            _dbg(debug, f"[ROW]#{seq} IGNORE bad balance {balance} -> use expected {expected}")
                            balance = expected
                            # note: do NOT set accepted_balance_cell when we override
                        else:
                            _dbg(debug, f"[ROW]#{seq} keep first-pass: resid {abs(resid):.2f} vs {ROW_RESID_TOL:.2f}")
                            accepted_balance_cell = True
                else:
                    # Already within tolerance
                    accepted_balance_cell = True
            else:
                # No prev or no balance — accept whatever we have (for totals we only need debit/credit)
                accepted_balance_cell = False

            # Build description (unchanged logic)
            df["center"] = df["left"] + (df["width"] / 2.0)
            details_span = df[(df["center"] >= details_left - 4) & (df["center"] <= debit_right - 4)].copy()

            parts: List[str] = []
            if not details_span.empty:
                for tok in details_span["text"].tolist():
                    t = (tok or "").strip()
                    if not t:
                        continue
                    if _is_currency_token(t) or _looks_like_amount_token(t):
                        continue
                    parts.append(t)
                if last_seen_date and parts:
                    variants = set(date_variants(last_seen_date))
                    parts = [p for p in parts if p not in variants]
                clean_desc = " ".join(parts).strip(" -|")
            else:
                clean_desc = (ltxt or "").strip()
                if last_seen_date:
                    for variant in date_variants(last_seen_date):
                        if variant in clean_desc:
                            clean_desc = clean_desc.replace(variant, "")
                for tok in [w.get("text", "") for w in words]:
                    tok = (tok or "").strip()
                    if not tok:
                        continue
                    if tok in {"|", "€", "£"} or _looks_like_amount_token(tok):
                        clean_desc = clean_desc.replace(tok, "")
                clean_desc = clean_desc.strip(" -|")

            all_transactions.append({
                "seq": seq,
                "transactions_date": last_seen_date,
                "transaction_type": "credit" if (credit or 0.0) > 0 else "debit",
                "description": clean_desc,
                "amount": float(credit if (credit or 0.0) > 0 else (debit or 0.0)),
            })

            # Update rolling balance for reconciliation only
            if balance is not None:
                prev_balance = float(balance)
                if accepted_balance_cell:
                    last_tx_balance = float(balance)  # only feed closing from an accepted cell

            seq += 1

    _dbg(debug, f"[TX] parsed={len(all_transactions)} last_tx_balance={last_tx_balance}")
    return all_transactions, last_tx_balance

# ----------------------------
# IBAN extraction & currency inference
# ----------------------------

def extract_iban(pages: list[dict]) -> str | None:
    for page in pages or []:
        for line in page.get("lines", []) or []:
            txt = (line.get("line_text") or "")
            up = txt.upper()
            if "IBAN:" in up and "(BIC" in up:
                try:
                    part = up.split("IBAN:", 1)[1].split("(BIC", 1)[0]
                    iban = re.sub(r"[^A-Z0-9]", "", part)
                    return iban if iban else None
                except Exception:
                    continue
    return None

def infer_currency_from_headers(pages: list[dict]) -> str | None:
    for page in pages or []:
        lines = page.get("lines", []) or []
        for line in lines:
            words = line.get("words", []) or []
            for i, w in enumerate(words):
                t = (w.get("text") or "").strip().lower()
                if t in ("debit", "credit", "balance") and i + 1 < len(words):
                    nxt = (words[i + 1].get("text") or "").strip()
                    if _is_currency_token(nxt):
                        u = nxt.strip().upper()
                        return "GBP" if u in ("£", "GBP") else "EUR"
    return None

# ----------------------------
# Build a single statement node
# ----------------------------

def _make_statement_node(raw_ocr: dict, client: str, account_type: str, debug: bool = True) -> dict:
    pages = raw_ocr.get("pages", []) or []

    iban = extract_iban(pages)
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    if not iban:
        inferred = infer_currency_from_headers(pages)
        if inferred:
            currency = inferred

    opening_val, start_date = extract_opening_balance_and_start_date(pages, debug=debug)
    _dbg(debug, f"[SUM] opening_val={opening_val} start={start_date}")

    transactions, last_tx_balance = parse_transactions(pages, iban=iban, opening_seed=opening_val, debug=debug)

    closing_val_page = extract_closing_balance(pages)
    if last_tx_balance is not None:
        closing_val = last_tx_balance
        _dbg(debug, f"[SUM] closing from last_tx_balance={closing_val}")
    else:
        closing_val = closing_val_page
        _dbg(debug, f"[SUM] closing from page-scan={closing_val}")

    money_in  = round(sum(t["amount"] for t in transactions if t["transaction_type"] == "credit"), 2)
    money_out = round(sum(t["amount"] for t in transactions if t["transaction_type"] == "debit"), 2)

    closing_calc = None
    if opening_val is not None:
        closing_calc = round(opening_val + money_in - money_out, 2)

    if transactions:
        all_dates = [t["transactions_date"] for t in transactions if t["transactions_date"]]
        start_date = start_date or (min(all_dates) if all_dates else None)
        end_date   = max(all_dates) if all_dates else None
    else:
        end_date = None

    currencies = {
        currency: {
            "balances": {
                "opening_balance": {
                    "summary_table": None,
                    "transactions_table": opening_val,
                },
                "money_in_total": {
                    "summary_table": None,
                    "transactions_table": money_in,
                },
                "money_out_total": {
                    "summary_table": None,
                    "transactions_table": money_out,
                },
                "closing_balance": {
                    "summary_table": None,
                    "transactions_table": closing_val,
                    "calculated": closing_calc,
                },
            },
            "transactions": transactions,
        }
    }

    sid_basis = f"{raw_ocr.get('file_name') or ''}|{start_date or ''}|{end_date or ''}"
    statement_id = hashlib.sha1(sid_basis.encode("utf-8")).hexdigest()[:12] if sid_basis.strip("|") else None

    return {
        "statement_id": statement_id,
        "file_name": raw_ocr.get("file_name"),
        "institution": "AIB",
        "account_type": account_type,
        "account_holder": None,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }

# ----------------------------
# Public entrypoints
# ----------------------------

def parse_statement_bundle(raw_ocr: dict, client: str = "Unknown", account_type: str = "Unknown", debug: bool = True) -> dict:
    stmt = _make_statement_node(raw_ocr, client=client, account_type=account_type, debug=debug)
    return {
        "schema_version": "bank-ocr.v1",
        "client": client,
        "statements": [stmt],
    }

def parse_statement(raw_ocr: dict, client: str = "Unknown", account_type: str = "Unknown", debug: bool = True) -> dict:
    return _make_statement_node(raw_ocr, client=client, account_type=account_type, debug=debug)
