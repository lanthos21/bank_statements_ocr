# parsers/ptsb.py
# PTSB parser (image-based PDF friendly)
# Uses header-centers to build true windows for Withdrawn / Paid In / Balance.
# Robust numeric extraction with:
#  - token-center windowing
#  - adjacent-token merge
#  - whole-window "cents join" (e.g. "2924 81" -> 2924.81)
#  - slash→7 repair (e.g. "15/30.16" -> 15730.16, "1/54.47" -> 1754.47)
#  - per-run cents heuristic for dotless reads (e.g. "1564163" -> 15641.63)
# Adds date carry-forward for rows whose date token drops in OCR.

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

from utils import parse_currency  # your existing helper

# ----------------------------
# Regexes / Date parsing
# ----------------------------

# Dates like "24APR24" or "24 APR 24" or "4May2024"
RE_DATE_PTSB = re.compile(r"\b(\d{1,2})\s*([A-Za-z]{3})\s*(\d{2,4})\b")
_MONTHS = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}
def _find_date_in_text_ptsb(s: str) -> Optional[str]:
    if not s:
        return None
    m = RE_DATE_PTSB.search(s)
    if not m:
        return None
    d, mon, y = m.groups()
    mon = (mon or "").upper()
    mm = _MONTHS.get(mon)
    if not mm:
        return None
    yy = y
    if len(yy) == 2:
        yy = "20" + yy
    try:
        return f"{int(yy):04d}-{int(mm):02d}-{int(d):02d}"
    except Exception:
        return None

# Special row labels
RE_BAL_FROM_LAST = re.compile(
    r"\b(Balance\s+from\s+last\s+stmt|Balance\s+brought\s+forward|Balance\s+b\/f)\b",
    re.IGNORECASE,
)
RE_CLOSING_BAL = re.compile(
    r"\b(Closing\s+Balance|Closing\s+bal|Balance\s+carried\s+forward|Closing\s+balance\s+carried\s+forward|Balance\s+c\/f)\b",
    re.IGNORECASE,
)

# ----------------------------
# Small geometry helpers
# ----------------------------

def _right(w: dict) -> int:
    return int(w.get("left", 0)) + int(w.get("width", 0))

def _center_x(w: dict) -> float:
    return int(w.get("left", 0)) + 0.5 * int(w.get("width", 0))

# ----------------------------
# Column window detection (centered headers)
# ----------------------------

def _detect_header_centers(lines: list[dict]) -> Optional[dict]:
    wd_cx = None
    pi_span = None  # (min_left, max_right) across 'Paid' and 'In'
    bal_cx = None

    for line in lines or []:
        words = line.get("words", []) or []
        n = len(words)
        for i, w in enumerate(words):
            t = (w.get("text") or "").strip().lower()
            if not t:
                continue

            if t == "withdrawn" and wd_cx is None:
                wd_cx = _center_x(w)

            if t == "paid":
                if i + 1 < n and (words[i+1].get("text") or "").strip().lower() == "in":
                    lefts = [int(w.get("left", 0)), int(words[i+1].get("left", 0))]
                    rights = [_right(w), _right(words[i+1])]
                    pi_span = (float(min(lefts)), float(max(rights)))
            elif t in ("paidin", "paid-in", "paidin.", "paid-in."):
                pi_span = (float(int(w.get("left", 0))), float(_right(w)))

            if t == "balance" and bal_cx is None:
                bal_cx = _center_x(w)

    if wd_cx is None or pi_span is None or bal_cx is None:
        return None

    paidin_cx = (pi_span[0] + pi_span[1]) * 0.5
    return {"withdrawn_cx": float(wd_cx), "paidin_cx": float(paidin_cx), "balance_cx": float(bal_cx)}

def _build_column_windows_from_centers(centers: dict) -> dict:
    cx_w = centers["withdrawn_cx"]
    cx_p = centers["paidin_cx"]
    cx_b = centers["balance_cx"]

    gap_wp = abs(cx_p - cx_w)
    gap_pb = abs(cx_b - cx_p)
    gap = min(gap_wp, gap_pb)

    HALF_FACTOR = 0.50  # tolerant but avoids info panel
    half = gap * HALF_FACTOR
    def win(cx: float) -> tuple[float, float]:
        return (cx - half, cx + half)

    return {
        "withdrawn_window": win(cx_w),
        "paidin_window":    win(cx_p),
        "balance_window":   win(cx_b),
    }

def detect_column_windows_ptsb(lines: list[dict]) -> Optional[tuple[dict, int, int]]:
    centers = _detect_header_centers(lines)
    if not centers:
        return None

    windows = _build_column_windows_from_centers(centers)

    hdr_idx = 0
    for i, line in enumerate(lines or []):
        txt = (line.get("line_text") or "").lower()
        if all(k in txt for k in ("date", "details")) and ("withdrawn" in txt or "paid" in txt or "balance" in txt):
            hdr_idx = i
            break

    return windows, hdr_idx, len(lines)

# ----------------------------
# Amount extraction in a window
# ----------------------------

_NUMLIKE = re.compile(r"[0-9.,\-€£$/]")
# --- FIX 1: safe replacement using \g<1>7\g<2> ---
def _slash_to_seven(s: str) -> str:
    # Replace digit '/' digit with digit '7' digit (e.g. 15/30.16 -> 15730.16)
    return re.sub(r"(\d)\s*/\s*(\d)", r"\g<1>7\g<2>", s)

# --- FIX 2: prefer the repaired candidate before the cleaned one ---
def _normalize_for_parse(s: str) -> list[str]:
    raw = s
    cleaned = re.sub(r"[^0-9,\.\-€£$]", "", raw)

    # Try in this order: raw → slash→7(raw) → cleaned → slash→7(cleaned)
    ordered = [raw, _slash_to_seven(raw), cleaned, _slash_to_seven(cleaned)]

    out = []
    seen = set()
    for base in ordered:
        for v in (base, base.replace("O", "0").replace("o", "0")):
            if v not in seen:
                out.append(v)
                seen.add(v)
    return out

def _best_amount_in_window(
    words: list[dict],
    window: tuple[float, float],
    pad: int = 18,
    merge_gap_px: int = 28,
    debug: bool = False,
    dbg=print,
    label: str = ""
) -> tuple[Optional[int], Optional[float]]:
    """
    Find best numeric in [window±pad] using token-center.
    - Merge adjacent tokens if gap ≤ merge_gap_px.
    - Try multiple parse candidates, including slash→7 repair.
    - Per-run cents heuristic for dotless integers (≥3 digits).
    - Whole-window "cents join" when no dot in any token and multiple runs exist.
    Returns (x_right_of_choice, value) or (None, None).
    """
    c0, c1 = float(window[0]) - pad, float(window[1]) + pad
    cand = [w for w in (words or [])
            if _NUMLIKE.search((w.get("text") or "")) and (c0 <= _center_x(w) <= c1)]
    if not cand:
        if debug and label:
            dbg(f"— {label}: window=({window[0]:.1f},{window[1]:.1f}) → no numeric tokens by center")
        return (None, None)

    cand.sort(key=lambda w: int(w.get("left", 0)))
    runs, cur, last_r = [], [], None
    for w in cand:
        l = int(w.get("left", 0)); r = _right(w)
        if last_r is None or l - last_r <= merge_gap_px:
            cur.append(w)
        else:
            if cur: runs.append(cur)
            cur = [w]
        last_r = r
    if cur: runs.append(cur)

    best = None  # (xr, float(value), src)

    # First pass: evaluate each run
    for run in runs:
        xr = max(_right(w) for w in run)
        raw_join = "".join((w.get("text") or "") for w in run).replace(" ", "")

        v, src, used_cents = None, "raw", False
        for cand_s in _normalize_for_parse(raw_join):
            v = parse_currency(cand_s, strip_currency=False)
            if v is not None:
                src = f"parsed:{cand_s!r}"
                break

        if v is None:
            digits = re.sub(r"[^\d\-]", "", raw_join)
            if re.fullmatch(r"-?\d{3,}", digits):
                sign = -1 if digits.startswith("-") else 1
                v = sign * (int(digits.lstrip("-")) / 100.0)
                used_cents = True
                src = "cents-heuristic(run)"

        if debug and label:
            dbg(f"   · {label} RUN raw={raw_join!r} → v={v!r} xr={xr} src={src}{' (USED)' if used_cents else ''}")

        if v is None:
            continue
        if best is None or xr > best[0]:
            best = (xr, float(v), src)

    # Second pass: if no dot/comma in any token and >1 run, attempt whole-window cents join
    any_dot = any("." in (w.get("text") or "") or "," in (w.get("text") or "") for w in cand)
    if not any_dot and len(runs) >= 2:
        all_digits = "".join(re.sub(r"[^\d]", "", (w.get("text") or "")) for w in cand)
        if re.fullmatch(r"\d{3,}", all_digits):
            xr = max(_right(w) for w in cand)
            v = int(all_digits) / 100.0
            if debug and label:
                dbg(f"   · {label} WHOLE-WINDOW cents-join digits={all_digits} → v={v:.2f} xr={xr}")
            if best is None or xr >= best[0]:
                best = (xr, float(v), "cents-join(window)")

    if not best:
        if debug and label:
            dbg(f"— {label}: no parseable runs")
        return (None, None)

    if debug and label:
        dbg(f"— {label}: BEST val={best[1]:.2f} xr={best[0]} src={best[2]}")
    return (best[0], best[1])

# ----------------------------
# Opening & Closing using labelled rows
# ----------------------------

def extract_opening_closing_ptsb(
    pages: list[dict],
    debug: bool = False,
    dbg=print
) -> tuple[float | None, float | None]:
    opening_value: Optional[float] = None
    closing_value: Optional[float] = None

    last_windows: Optional[dict] = None

    for pidx, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        hdr = detect_column_windows_ptsb(lines)
        if hdr:
            last_windows = hdr[0]

        if not last_windows:
            continue

        bal_win = last_windows["balance_window"]

        for line in lines:
            ltxt = (line.get("line_text") or "")
            words = line.get("words", []) or []

            if RE_BAL_FROM_LAST.search(ltxt) and opening_value is None:
                _, val = _best_amount_in_window(words, bal_win, debug=debug, dbg=dbg, label="OPENING balance")
                if val is not None:
                    opening_value = float(val)

            if RE_CLOSING_BAL.search(ltxt):
                _, val = _best_amount_in_window(words, bal_win, debug=debug, dbg=dbg, label="CLOSING balance")
                if val is not None:
                    closing_value = float(val)  # last wins

    return opening_value, closing_value

# ----------------------------
# Transactions (PTSB)
# ----------------------------

def parse_transactions_ptsb(
    pages: list[dict],
    iban: str | None = None,
    debug: bool = False,
    dbg=print,
    focus_terms: list[str] | None = None
) -> tuple[list[dict], list[dict]]:
    """
    Parse all PTSB transaction rows across pages.
    - Date carry-forward: if line has numbers but date OCR is missing, inherit the last seen date.
    - Skip special rows.
    - Extract debit/credit/balance by windowed token-center logic with robust numeric parsing.
    Returns (transactions, debug_rows)
    """
    all_transactions: List[dict] = []
    debug_rows: List[dict] = []

    current_currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    seq = 0
    last_windows: Optional[dict] = None
    last_date: Optional[str] = None  # carry-forward date
    WINDOW_PAD = 18
    MERGE_GAP = 28

    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_windows_ptsb(lines)

        if header:
            windows, start_idx, end_idx = header
            last_windows = windows
            line_start, line_end = start_idx + 1, end_idx
            if debug:
                dbg(f"\n📐 P{page_num} windows:"
                    f"  W[{windows['withdrawn_window'][0]:.1f},{windows['withdrawn_window'][1]:.1f}]"
                    f"  P[{windows['paidin_window'][0]:.1f},{windows['paidin_window'][1]:.1f}]"
                    f"  B[{windows['balance_window'][0]:.1f},{windows['balance_window'][1]:.1f}]")
        else:
            if last_windows is None:
                continue
            windows = last_windows
            line_start, line_end = 0, len(lines)

        for line in lines[line_start: line_end]:
            words = line.get("words", []) or []
            if not words:
                continue

            ltxt = (line.get("line_text") or "")
            y = int(words[0].get("top", 0))

            # Skip labelled/synthetic rows
            if RE_BAL_FROM_LAST.search(ltxt) or RE_CLOSING_BAL.search(ltxt):
                continue

            # Try to get a date; if missing, we'll carry-forward later if amounts exist
            tdate = _find_date_in_text_ptsb(ltxt)
            if tdate is not None:
                last_date = tdate

            focus = (not focus_terms) or any(k.lower() in ltxt.lower() for k in (focus_terms or []))
            if debug and focus and tdate:
                dbg(f"\nP{page_num} y≈{y} {tdate}  {ltxt!r}")
            elif debug and focus and not tdate:
                dbg(f"\nP{page_num} y≈{y} —no-date—  {ltxt!r}")

            # Extract amounts
            _, debit  = _best_amount_in_window(words, windows["withdrawn_window"], pad=WINDOW_PAD, merge_gap_px=MERGE_GAP, debug=debug and focus, dbg=dbg, label="Withdrawn")
            _, credit = _best_amount_in_window(words, windows["paidin_window"],    pad=WINDOW_PAD, merge_gap_px=MERGE_GAP, debug=debug and focus, dbg=dbg, label="Paid In")
            _, bal    = _best_amount_in_window(words, windows["balance_window"],   pad=WINDOW_PAD, merge_gap_px=MERGE_GAP, debug=debug and focus, dbg=dbg, label="Balance")

            # If we have no numeric in any window, skip (header/separator/info)
            if (debit is None or debit == 0.0) and (credit is None or credit == 0.0) and bal is None:
                continue

            # If OCR dropped the date on this row, carry-forward the last seen date
            row_date = tdate if tdate is not None else last_date
            if row_date is None:
                # Still no date → too risky to keep
                if debug and focus:
                    dbg("   · decision: SKIP row (no date and no carry-forward available)")
                continue

            # Clean description: remove the first matched date token (if any)
            clean_desc = ltxt
            dm = RE_DATE_PTSB.search(clean_desc)
            if dm:
                clean_desc = clean_desc.replace(dm.group(0), "").strip(" |-")

            tx_type = "credit" if (credit or 0.0) > 0 else "debit"
            amt_val = float(credit if (credit or 0.0) > 0 else (debit or 0.0))

            tx = {
                "seq": seq,
                "transactions_date": row_date,
                "transaction_type": tx_type,
                "description": clean_desc,
                "amount": {
                    "value": amt_val,
                    "currency": current_currency,
                },
                "balance_after_statement": None if bal is None else {
                    "value": float(bal),
                    "currency": current_currency,
                },
                "balance_after_calculated": None,
            }
            all_transactions.append(tx)
            seq += 1

            debug_rows.append({
                "page": page_num,
                "y": y,
                "date": row_date,
                "line": ltxt,
                "debit": float(debit) if debit is not None else None,
                "credit": float(credit) if credit is not None else None,
                "balance": float(bal) if bal is not None else None,
            })

            if debug and focus:
                dbg(f"   · decision: KEEP  debit={debit if debit is not None else '—'}  credit={credit if credit is not None else '—'}  balance={bal if bal is not None else '—'}")

    return all_transactions, debug_rows

# ----------------------------
# Build multi-currency sections (from rows)
# ----------------------------

def _group_by_currency(transactions: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in transactions:
        cur = t.get("amount", {}).get("currency")
        if cur:
            buckets[cur].append(t)
    for cur in buckets:
        buckets[cur].sort(key=lambda t: t.get("seq", 0))  # preserve order
    return buckets

def _derive_open_close_from_rows(txs: List[dict]) -> tuple[float | None, float | None]:
    if not txs:
        return None, None
    first, last = txs[0], txs[-1]

    first_bal = (first.get("balance_after_statement") or {}).get("value")
    first_amt = first.get("amount", {}).get("value")
    first_typ = first.get("transaction_type")

    opening = None
    if first_bal is not None and first_amt is not None and first_typ in ("credit", "debit"):
        if first_typ == "credit":
            opening = round(float(first_bal) - float(first_amt), 2)
        else:
            opening = round(float(first_bal) + float(first_amt), 2)

    closing_stmt = (last.get("balance_after_statement") or {}).get("value")
    closing_stmt = None if closing_stmt is None else float(closing_stmt)

    return opening, closing_stmt

def _build_currency_sections_from_rows(buckets: Dict[str, List[dict]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for cur in sorted(buckets.keys()):
        txs = buckets[cur]

        money_in  = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit"), 2)

        opening_from_rows, closing_stmt_from_rows = _derive_open_close_from_rows(txs)

        closing_calc = None
        if opening_from_rows is not None:
            closing_calc = round(float(opening_from_rows) + money_in - money_out, 2)

        out[cur] = {
            "opening_balance": None if opening_from_rows is None else {"value": opening_from_rows, "currency": cur},
            "money_in_total":  {"value": money_in,  "currency": cur},
            "money_out_total": {"value": money_out, "currency": cur},
            "closing_balance_statement": None if closing_stmt_from_rows is None else {"value": closing_stmt_from_rows, "currency": cur},
            "closing_balance_calculated": None if closing_calc is None else {"value": closing_calc, "currency": cur},
            "transactions": txs,
        }

    return out

# ----------------------------
# IBAN extraction
# ----------------------------

def extract_iban(pages: list[dict]) -> str | None:
    iban_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b')
    for page in pages:
        for line in page.get("lines", []):
            txt = line.get("line_text", "") or ""
            if "IBAN" in txt.upper():
                after = txt.upper().split("IBAN", 1)[-1]
                cleaned = re.sub(r"\s+", "", after)
                m = iban_pattern.search(cleaned)
                if m:
                    return m.group(1)
    return None

# ----------------------------
# Public entrypoint
# ----------------------------

def parse_statement(raw_ocr, client="Unknown", account_type="Unknown", debug: bool = True):
    def dbg(*a):
        if debug:
            print(*a)

    pages = raw_ocr.get("pages", []) or []

    # IBAN & currency hint
    iban = extract_iban(pages)
    currency_hint = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    # Transactions (with debug rows)
    transactions, debug_rows = parse_transactions_ptsb(
        pages, iban=iban, debug=debug, dbg=dbg, focus_terms=["5.40","APCOA","Q PARK","€5.40","5,40"]  # set to ["APCOA","5.40"] if you want to zoom logs
    )
    dbg(f"🧾 Parsed {len(transactions)} transactions")

    # Group by currency & per-currency sections
    buckets = _group_by_currency(transactions)
    dbg("🔑 Buckets from rows:", list(buckets.keys()))
    currencies = _build_currency_sections_from_rows(buckets)
    for k, sec0 in currencies.items():
        dbg(f"   - {repr(k)} BEFORE overlay: open_from_rows={sec0.get('opening_balance')} "
            f"in={sec0.get('money_in_total')} out={sec0.get('money_out_total')}")

    # Explicit opening/closing using labelled rows (robust)
    opening_val, closing_val = extract_opening_closing_ptsb(pages, debug=debug, dbg=dbg)
    dbg(f"📘 Explicit opening_val={opening_val}")
    dbg(f"📘 Explicit closing_val={closing_val}")

    # Choose target currency
    target_cur = "EUR" if "EUR" in currencies else next(iter(currencies.keys()), currency_hint)
    dbg("🎯 target_cur for overlay:", repr(target_cur))

    if target_cur not in currencies:
        currencies[target_cur] = {
            "opening_balance": None,
            "money_in_total": {"value": 0.0, "currency": target_cur},
            "money_out_total": {"value": 0.0, "currency": target_cur},
            "closing_balance_statement": None,
            "closing_balance_calculated": None,
            "transactions": [],
        }
        dbg(f"ℹ️ Created empty bucket for {repr(target_cur)}")

    sec = currencies[target_cur]

    if opening_val is not None:
        sec["opening_balance"] = {"value": float(round(opening_val, 2)), "currency": target_cur}
        dbg(f"✅ Overlaid OPENING on {repr(target_cur)}:", sec["opening_balance"])
    else:
        dbg("⚠️ No explicit opening to overlay")

    if closing_val is not None:
        sec["closing_balance_statement"] = {"value": float(round(closing_val, 2)), "currency": target_cur}
        dbg(f"✅ Overlaid CLOSING on {repr(target_cur)}:", sec["closing_balance_statement"])
    else:
        dbg("⚠️ No explicit closing to overlay")

    # Recompute calculated closing if we have opening + totals
    if sec.get("opening_balance") and sec.get("money_in_total") and sec.get("money_out_total"):
        o = float(sec["opening_balance"]["value"])
        inm = float(sec["money_in_total"]["value"])
        outm = float(sec["money_out_total"]["value"])
        sec["closing_balance_calculated"] = {"value": round(o + inm - outm, 2), "currency": target_cur}
        dbg("🧮 closing_balance_calculated:", sec["closing_balance_calculated"])

    # Statement dates from transactions
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = None
        end_date = None

    # Optional: write a quick CSV of row decisions to help spot misses
    if debug and debug_rows:
        try:
            import os, csv
            os.makedirs("results", exist_ok=True)
            dbg_csv = os.path.join("results", "ptsb_debug_rows.csv")
            with open(dbg_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["page","y","date","line","debit","credit","balance"])
                w.writeheader()
                w.writerows(debug_rows)
            print(f"🧪 PTSB debug rows saved to {dbg_csv}")
        except Exception as e:
            print(f"⚠️ Failed to write ptsb_debug_rows.csv: {e}")

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,
        "institution": "PTSB",
        "account_type": account_type,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
