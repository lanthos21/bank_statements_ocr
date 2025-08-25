# parsers/ptsb.py
# PTSB parser (image-based PDF friendly) — stable recall + cleaned descriptions
# - Uses header-centers to build real windows for Withdrawn / Paid In / Balance
# - Robust numeric extraction with configurable guards:
#     * token-center windowing
#     * adjacent-token merge
#     * slash→7 repair (e.g. "15/30.16" → 15730.16)
#     * optional cents heuristic (disabled for per-row amounts to avoid false positives)
#     * whole-window cents-join if multiple runs w/o separators
# - Description = words strictly LEFT of the amount columns (drops amounts & sidebar)
# - Drops page furniture and “Balance B/fwd / from last stmt” lines (not transactions)
# - Opening balance from explicit “Balance from last stmt …” (with repairs + rescue)
# - Closing balance from explicit row, else last transaction’s Balance cell (with rescue)
# - Strict balances schema, amount is a float in each tx

from __future__ import annotations

import hashlib
import re
from typing import Optional, Dict, List, Tuple, Any
# --- put near your other imports ---
from typing import Optional

import pandas as pd
import cv2
import pytesseract
import numpy as np

# ---------------------------------
# Small parse helpers
# ---------------------------------
def parse_currency(s: str, strip_currency: bool = False) -> Optional[float]:
    """
    Lightweight resilient currency parser.
    Accepts "1,234.56", "1.234,56", "+123", "-123", "123€", etc.
    Returns float or None.
    """
    if s is None:
        return None
    t = str(s).replace("−", "-")
    m = re.search(r"[-+]?\d[\d,.\s]*\d|\d", t)
    if not m:
        return None
    frag = (
        m.group(0)
        .replace("\u00A0", " ")
        .replace("\u2009", " ")
        .replace("\u202F", " ")
        .strip()
    )

    sign = ""
    if frag.startswith(("+", "-")):
        sign, frag = frag[0], frag[1:]

    if "," in frag and "." in frag:
        last_comma = frag.rfind(",")
        last_dot   = frag.rfind(".")
        if last_comma > last_dot:
            frag = frag.replace(".", "").replace(",", ".")
        else:
            frag = frag.replace(",", "")
    else:
        if frag.count(",") == 1 and frag.count(".") == 0:
            frag = frag.replace(",", ".")
        else:
            frag = frag.replace(",", "")

    try:
        return float(sign + frag)
    except Exception:
        return None


# ---------------------------------
# Regexes / Date parsing
# ---------------------------------
RE_DATE_PTSB = re.compile(r"\b(\d{1,2})\s*([A-Za-z]{3})\s*(\d{2,4})\b")
RE_COMPACT_DATE_TOKEN = re.compile(r"\b\d{1,2}[A-Za-z]{3}\d{2}\b")

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

def _find_compact_date_token(s: str) -> Optional[str]:
    m = RE_COMPACT_DATE_TOKEN.search((s or ""))
    return m.group(0) if m else None


# Special rows and furniture
RE_BAL_FROM_LAST = re.compile(
    r"\b(?:Balance\s+from\s+last\s+stmt|Balance\s+brought\s+forward|Balance\s+b\/f|Balance\s+B\/fwd)\b",
    re.IGNORECASE,
)
RE_CLOSING_BAL = re.compile(
    r"\b(?:Closing\s+Balance|Closing\s+bal|Balance\s+carried\s+forward|Closing\s+balance\s+carried\s+forward|Balance\s+c\/f)\b",
    re.IGNORECASE,
)
RE_PAGE_FURNITURE = re.compile(
    r"(Date\s+Details\s+Withdrawn|Statement\s+Number|Account\s+Number|IBAN|www\.permanenttsb\.ie|Overdraft\s+In\s+Details|Available\s+Balance)",
    re.IGNORECASE,
)

# ---------------------------------
# Small geometry helpers
# ---------------------------------
def _right(w: dict) -> int:
    return int(w.get("left", 0)) + int(w.get("width", 0))

def _center_x(w: dict) -> float:
    return int(w.get("left", 0)) + 0.5 * int(w.get("width", 0))

def _norm(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("\u00A0", " ")
        .replace("\u2009", " ")
        .replace("\u202F", " ")
        .replace("−", "-")
        .strip()
    )


# ---------------------------------
# OCR second pass helpers
# ---------------------------------

def _numeric_ocr_roi(
    img_path: str,
    x0: int, x1: int, y0: int, y1: int,
    pad: int = 6,
    require_decimal: bool = True,
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
    raw = (pytesseract.image_to_string(roi, config=config) or "").strip().replace("−", "-").replace(" ", "")
    if require_decimal and (("." not in raw) and ("," not in raw)):
        return None
    return parse_currency(raw, strip_currency=False)

# Characters we accept in a clean amount capture


# --- helpers / gates ---

_ALLOWED_AMOUNT_CHARS = set("0123456789.,+-€£$()")

def _normalize_ws_and_minus(s: str) -> str:
    return (
        (s or "")
        .replace("\u00A0", " ")
        .replace("\u2009", " ")
        .replace("\u202F", " ")
        .replace("−", "-")  # unicode minus → hyphen-minus
    )

def _window_text(words: list[dict], window: tuple[float, float], pad: int = 10) -> str:
    c0, c1 = float(window[0]) - pad, float(window[1]) + pad
    toks = []
    for w in words or []:
        L = int(w.get("left", 0)); W = int(w.get("width", 0))
        cx = L + 0.5 * W
        if c0 <= cx <= c1:
            t = (w.get("text") or "")
            # keep tokens that at least look amount-like
            if any(ch.isdigit() for ch in t) or re.search(r"[.,+-]", t):
                toks.append(t)
    return "".join(toks)

def _should_run_roi(win_str: str, val_win_is_none: bool) -> bool:
    """
    Run ROI if:
      - first pass failed AND there is at least one digit, OR
      - digits are present AND we see a suspicious char/pattern (/, O/o, I/l, trailing slash).
    Never run if there are no digits.
    """
    s = _normalize_ws_and_minus(win_str)
    has_digit = any(ch.isdigit() for ch in s)
    if not has_digit:
        return False
    if val_win_is_none:
        return True
    if any(c in s for c in "/OoIl"):
        return True
    if re.search(r"[.,]\s*/|/$|/\s*\d$", s):  # e.g. '52937.2/' or '59/20./4'
        return True
    # otherwise accept first pass
    return False

def _rightmost_num_run_bounds(
    words: list[dict],
    window: tuple[float, float],
    pad: int = 10,
    merge_gap_px: int = 28
) -> tuple[Optional[int], Optional[int]]:
    """Tight X-bounds around the rightmost digit-containing run inside the window."""
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

    run = runs[-1]  # rightmost
    x0 = min(int(w.get("left", 0)) for w in run)
    x1 = max(int(w.get("left", 0)) + int(w.get("width", 0)) for w in run)
    return (x0, x1)

# --- the fixed function (no ellipses) ---

def _amount_with_numeric_fallback(
    words: list[dict],
    window: tuple[float, float],
    page_img_path: Optional[str],
    line_y0: Optional[int],
    line_y1: Optional[int],
    pad: int,
    merge_gap_px: int,
    require_decimal: bool,
) -> Optional[float]:
    # 1) fast token pass
    _, val_win = _best_amount_in_window(
        words, window,
        pad=pad, merge_gap_px=merge_gap_px,
        require_decimal=require_decimal, allow_cents_heuristic=False
    )
    if val_win is not None:
        val_win = float(val_win)

    # 2) inspect raw text in the window
    win_str = _window_text(words, window, pad=pad)
    need_roi = _should_run_roi(win_str, val_win_is_none=(val_win is None))
    if not need_roi or not page_img_path or line_y0 is None or line_y1 is None:
        return val_win  # blank/clean → keep first pass (or None)

    # 3) tight crop around rightmost numeric run (fallback to right-edge band)
    rx0, rx1 = _rightmost_num_run_bounds(words, window, pad=pad, merge_gap_px=merge_gap_px)
    if rx0 is not None and rx1 is not None:
        x0, x1 = rx0 - 4, rx1 + 4
    else:
        w0, w1 = int(window[0]), int(window[1])
        width = w1 - w0
        band = max(int(width * 0.5), 140)
        x1 = w1 + 6
        x0 = max(w0 - 6, x1 - band)

    val_roi = _numeric_ocr_roi(page_img_path, x0, x1, line_y0, line_y1, pad=0, require_decimal=require_decimal)

    # 4) accept ROI cautiously; prefer it only when suspicion is real (we already checked)
    if val_roi is not None and val_win is not None:
        # avoid wild magnitude jumps unless raw had '/'
        if ("/" in win_str) or (0.1 <= (val_roi / max(val_win, 1e-6)) <= 10.0):
            print(f"Running a 2nd OCR to get {val_roi} from {val_win}  raw='{win_str}'")
            return float(val_roi)
        return val_win

    if val_roi is not None:
        print(f"Running a 2nd OCR to get {val_roi} from {val_win}  raw='{win_str}'")
    return float(val_roi) if val_roi is not None else val_win


def _cents_join_from_window_tokens(
    words: list[dict],
    window: tuple[float, float],
    pad: int = 10,
) -> Optional[float]:
    """
    If the balance cell is rendered as '2770 34' (euros + cents as separate tokens),
    join all digits and divide by 100. Only applies when there is no ',' or '.' in the window
    and the rightmost digit run has exactly 2 digits (cents).
    """
    c0, c1 = float(window[0]) - pad, float(window[1]) + pad
    tokens: list[str] = []
    digit_runs: list[str] = []

    for w in (words or []):
        t = (w.get("text") or "")
        if not t:
            continue
        cx = int(w.get("left", 0)) + 0.5 * int(w.get("width", 0))
        if not (c0 <= cx <= c1):
            continue
        tokens.append(t)
        digits = re.sub(r"[^\d]", "", t)
        if digits:
            digit_runs.append(digits)

    if not tokens or not digit_runs:
        return None

    win_str = "".join(tokens)
    if "." in win_str or "," in win_str:
        return None  # not a space-cents case

    # need at least two digit runs, and the last run should be 2 digits (the cents)
    if len(digit_runs) >= 2 and len(digit_runs[-1]) == 2:
        all_digits = "".join(digit_runs)
        if re.fullmatch(r"\d{3,}", all_digits):
            return float(int(all_digits) / 100.0)

    return None

# ---------------------------------
# Build a DataFrame of all words
# ---------------------------------
def _page_words_to_df(page: dict) -> pd.DataFrame:
    rows = []
    for ln in page.get("lines", []) or []:
        for w in ln.get("words", []) or []:
            rows.append(w)
    if not rows:
        return pd.DataFrame(columns=["text","left","top","width","height","right","bottom","cy","block_num","par_num","line_num"])
    df = pd.DataFrame(rows).copy()
    for c in ("left", "top", "width", "height"):
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)
    df["right"]  = df["left"] + df["width"]
    df["bottom"] = df["top"]  + df["height"]
    df["cy"]     = df["top"] + df["height"] * 0.5
    df["text"]   = df["text"].astype(str)
    for c in ("block_num", "par_num", "line_num"):
        if c not in df.columns:
            df[c] = -1
    return df

def _med_height(df: pd.DataFrame) -> float:
    return float(df["height"].median()) if ("height" in df and not df["height"].empty) else 18.0


# ---------------------------------
# Header detection → column windows
# ---------------------------------
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
    cx_w = centers["withdrawn_cx"]; cx_p = centers["paidin_cx"]; cx_b = centers["balance_cx"]
    gap_wp = abs(cx_p - cx_w); gap_pb = abs(cx_b - cx_p); gap = min(gap_wp, gap_pb)
    half = gap * 0.50  # tolerant but avoids sidebox
    def win(cx: float) -> tuple[float, float]: return (cx - half, cx + half)
    return {"withdrawn_window": win(cx_w), "paidin_window": win(cx_p), "balance_window": win(cx_b)}

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


# ---------------------------------
# Numeric extraction (configurable)
# ---------------------------------
_NUMLIKE = re.compile(r"[0-9.,\-€£$/]")

def _slash_to_seven(s: str) -> str:
    # Replace digit '/' digit with '7' →  15/30.16 -> 15730.16
    return s
    return re.sub(r"(\d)\s*/\s*(\d)", r"\g<1>7\g<2>", s)

def _normalize_for_parse(s: str) -> list[str]:
    raw = s
    cleaned = re.sub(r"[^0-9,\.\-€£$]", "", raw)
    ordered = [raw, _slash_to_seven(raw), cleaned, _slash_to_seven(cleaned)]
    out, seen = [], set()
    for base in ordered:
        for v in (base, base.replace("O", "0").replace("o", "0")):
            if v not in seen:
                out.append(v); seen.add(v)
    return out

def _parse_joined_amount(
    raw_join: str,
    require_decimal: bool,
    allow_cents_heuristic: bool,
) -> Optional[float]:
    """
    Prefer repaired/cleaned candidates first so cases like '15/30.16' resolve to 15730.16.
    Also skip candidates that lack a visible decimal when require_decimal=True.
    """
    base = raw_join or ""
    cleaned = re.sub(r"[^0-9,\.\-€£$]", "", base)

    # Prefer repairs first (critical: try slash→7 before the raw form)
    candidates: list[str] = []
    if "/" in base:
        candidates += [_slash_to_seven(base)]
    if "/" in cleaned:
        candidates += [_slash_to_seven(cleaned)]
    candidates += [cleaned, base]

    # Add O→0 variants, keep unique order
    ordered: list[str] = []
    seen = set()
    for c in candidates:
        for v in (c, c.replace("O", "0").replace("o", "0")):
            if v not in seen:
                seen.add(v)
                ordered.append(v)

    # Parse in order; enforce visible decimal on the candidate (not just the fragment)
    for cand_s in ordered:
        if require_decimal and (("." not in cand_s) and ("," not in cand_s)):
            continue
        v = parse_currency(cand_s, strip_currency=False)
        if v is not None:
            return float(v)

    # Last resort: cents heuristic
    if allow_cents_heuristic:
        digits = re.sub(r"[^\d\-]", "", base)
        if re.fullmatch(r"-?\d{3,}", digits):
            sign = -1 if digits.startswith("-") else 1
            return sign * (int(digits.lstrip("-")) / 100.0)

    return None


def _best_amount_in_window(
    words: list[dict],
    window: tuple[float, float],
    pad: int = 18,
    merge_gap_px: int = 28,
    require_decimal: bool = True,
    allow_cents_heuristic: bool = False,
) -> tuple[Optional[int], Optional[float]]:
    """
    Pick the rightmost parseable value within a column window.
    Guards:
      - require_decimal=True rejects dotless candidates (prevents giant false numbers)
      - allow_cents_heuristic controls whether we recover dotless merges
    """
    c0, c1 = float(window[0]) - pad, float(window[1]) + pad
    cand = [w for w in (words or []) if _NUMLIKE.search((w.get("text") or "")) and (c0 <= _center_x(w) <= c1)]
    if not cand:
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

    best = None  # (xr, value)
    for run in runs:
        xr = max(_right(w) for w in run)
        raw_join = "".join((w.get("text") or "") for w in run).replace(" ", "")
        # Skip pure slashy IDs with no decimal
        if "/" in raw_join and "." not in raw_join and "," not in raw_join and not allow_cents_heuristic:
            continue
        v = _parse_joined_amount(raw_join, require_decimal=require_decimal, allow_cents_heuristic=allow_cents_heuristic)
        if v is None:
            continue
        if best is None or xr > best[0]:
            best = (xr, float(v))

    # Whole-window cents-join: only if allowed (opening/closing path), not for per-row tx
    if best is None and allow_cents_heuristic:
        any_dot = any("." in (w.get("text") or "") or "," in (w.get("text") or "") for w in cand)
        if not any_dot and len(runs) >= 2:
            all_digits = "".join(re.sub(r"[^\d]", "", (w.get("text") or "")) for w in cand)
            if re.fullmatch(r"\d{3,}", all_digits):
                xr = max(_right(w) for w in cand)
                v = int(all_digits) / 100.0
                best = (xr, float(v))

    return best if best else (None, None)

def _best_amount_in_region_right_of(
    words: list[dict],
    x_min: int,
    x_max: Optional[int] = None,
    merge_gap_px: int = 28,
    require_decimal: bool = True,
    allow_cents_heuristic: bool = True,
) -> tuple[Optional[int], Optional[float]]:
    """
    Rescue: pick the rightmost amount in a horizontal band (x > x_min and x < x_max if given).
    Run-based like the window picker; honors decimal/heuristic guards.
    """
    cand = []
    for w in (words or []):
        L = int(w.get("left", 0)); R = _right(w)
        cx = L + (R - L) * 0.5
        if cx <= int(x_min):
            continue
        if x_max is not None and cx >= int(x_max):
            continue
        if _NUMLIKE.search((w.get("text") or "")):
            cand.append(w)
    if not cand:
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

    best = None
    for run in runs:
        xr = max(_right(w) for w in run)
        raw_join = "".join((w.get("text") or "") for w in run).replace(" ", "")
        if "/" in raw_join and "." not in raw_join and "," not in raw_join and not allow_cents_heuristic:
            continue
        v = _parse_joined_amount(raw_join, require_decimal=require_decimal, allow_cents_heuristic=allow_cents_heuristic)
        if v is None:
            continue
        if best is None or xr > best[0]:
            best = (xr, float(v))
    return best if best else (None, None)


# ---------------------------------
# Opening & Closing (with rescue)
# ---------------------------------
def extract_opening_closing_ptsb(
    pages: list[dict],
    debug: bool = False,
) -> tuple[float | None, float | None]:
    opening_value: Optional[float] = None
    closing_value: Optional[float] = None
    last_windows: Optional[dict] = None
    paidin_cx = None
    balance_right = None

    for page in (pages or []):
        lines = page.get("lines", []) or []
        hdr = detect_column_windows_ptsb(lines)
        if hdr:
            last_windows = hdr[0]
            paidin_cx = (last_windows["paidin_window"][0] + last_windows["paidin_window"][1]) * 0.5
            balance_right = last_windows["balance_window"][1]

        if not last_windows:
            continue

        rescue_xmin = None if paidin_cx is None else int(paidin_cx + 0.20 * (last_windows["balance_window"][0] - paidin_cx))
        rescue_xmax = None if balance_right is None else int(balance_right + 24)

        bal_win = last_windows["balance_window"]
        page_img_path = page.get("raster_path")

        for line in lines:
            ltxt = (line.get("line_text") or "")
            words = line.get("words", []) or []
            y0, y1 = line.get("y0"), line.get("y1")

            # --- OPENING ---
            if RE_BAL_FROM_LAST.search(ltxt) and opening_value is None:
                _, val_win = _best_amount_in_window(
                    words, bal_win,
                    require_decimal=True, allow_cents_heuristic=True
                )
                win_str = _window_text(words, bal_win, pad=10)
                cj = _cents_join_from_window_tokens(words, bal_win, pad=10)

                need_roi = _should_run_roi(win_str, val_win_is_none=(val_win is None))
                val_roi = None
                if need_roi and page_img_path and y0 is not None and y1 is not None:
                    x0, x1 = int(bal_win[0]) - 10, int(bal_win[1]) + 10
                    val_roi = _numeric_ocr_roi(page_img_path, x0, x1, y0, y1, pad=6, require_decimal=True)

                opening_value = (
                    float(cj) if cj is not None else
                    (float(val_roi) if val_roi is not None else
                     (float(val_win) if val_win is not None else opening_value))
                )


            # --- CLOSING ---  (elif to prevent any overlap with opening)
            elif RE_CLOSING_BAL.search(ltxt):
                # 1) window pass
                _, val_win = _best_amount_in_window(
                    words, bal_win,
                    require_decimal=True, allow_cents_heuristic=True
                )
                win_str = _window_text(words, bal_win, pad=10)

                # 2) SPECIAL: space-cents join (e.g., '2770 34' -> 2770.34)
                cj = _cents_join_from_window_tokens(words, bal_win, pad=10)

                # 3) decide if we need ROI (non-whitelist chars or first pass failed)
                need_roi = _should_run_roi(win_str, val_win_is_none=(val_win is None))
                val_roi = None
                if need_roi and page_img_path and y0 is not None and y1 is not None:
                    x0, x1 = int(bal_win[0]) - 10, int(bal_win[1]) + 10
                    val_roi = _numeric_ocr_roi(page_img_path, x0, x1, y0, y1, pad=6, require_decimal=True)

                # 4) pick the best candidate (prefer space-cents if present)
                closing_value = (
                    float(cj) if cj is not None else
                    (float(val_roi) if val_roi is not None else
                     (float(val_win) if val_win is not None else closing_value))
                )

    return opening_value, closing_value


# ---------------------------------
# Description building (left of amounts only)
# ---------------------------------
RE_AMOUNT_TOKEN  = re.compile(r"^[\+\-]?\d[\d,\.]*\d$")   # e.g., 20.50, 1,234.56
RE_JUNK_TRAIL    = re.compile(r"(This\s+is\s+an\s+eligible.*)$", re.IGNORECASE)
RE_LONG_DIGITS   = re.compile(r"\b\d{6,}\b")              # account-ish tails

def _description_from_words(words: list[dict], windows: dict, raw_compact_date: Optional[str]) -> str:
    if not words:
        return ""
    first_amt_left = min(float(windows["withdrawn_window"][0]), float(windows["paidin_window"][0]))
    kept = []
    for w in words:
        txt = _norm(w.get("text", "") or "")
        if not txt:
            continue
        if float(w.get("left", 0)) < first_amt_left - 4:
            kept.append(txt)

    s = " ".join(kept).strip()
    if not s:
        return ""

    # Remove leading compact date token like "30APR24" (with optional pipe)
    if raw_compact_date:
        s = re.sub(rf"^\s*{re.escape(raw_compact_date)}\s*\|?\s*", "", s)
    s = re.sub(r"^\s*\d{1,2}[A-Za-z]{3}\d{2}\s*\|?\s*", "", s)

    # Drop stray amounts / long digit runs / sidebar tails
    s = " ".join(t for t in s.split() if not RE_AMOUNT_TOKEN.match(t))
    s = RE_LONG_DIGITS.sub("", s)
    s = RE_JUNK_TRAIL.sub("", s)

    s = re.sub(r"\s{2,}", " ", s).strip(" -|")
    return s


# ---------------------------------
# Transactions
# ---------------------------------
def parse_transactions_ptsb(
    pages: list[dict],
    iban: str | None = None,
) -> tuple[list[dict], list[dict], float | None]:
    """
    Parse all transaction rows across pages.
    - Must have a date and either Withdrawn or Paid In (skip balance-only/furniture)
    - Per-row values: require decimal; cents heuristic disabled (prevents huge ID amounts)
    - Descriptions from left of amount columns, cleaned
    Returns (transactions, debug_rows, last_balance_seen)
    """
    all_transactions: List[dict] = []
    debug_rows: List[dict] = []

    current_currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    seq = 0
    last_windows: Optional[dict] = None
    last_date: Optional[str] = None
    WINDOW_PAD = 18
    MERGE_GAP = 28

    last_balance_seen: Optional[float] = None

    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_windows_ptsb(lines)

        if header:
            windows, start_idx, end_idx = header
            last_windows = windows
            line_start, line_end = start_idx + 1, end_idx
        else:
            if last_windows is None:
                continue
            windows = last_windows
            line_start, line_end = 0, len(lines)

        # NEW: get per-page processed raster path for ROI crops
        page_img_path = page.get("raster_path")

        for line in lines[line_start: line_end]:
            words = line.get("words", []) or []
            if not words:
                continue
            ltxt = (line.get("line_text") or "")

            # Skip furniture & labeled non-transaction rows
            if RE_PAGE_FURNITURE.search(ltxt):
                continue
            if RE_BAL_FROM_LAST.search(ltxt) or RE_CLOSING_BAL.search(ltxt):
                continue

            # Date gating
            tdate = _find_date_in_text_ptsb(ltxt)
            if tdate is not None:
                last_date = tdate
            row_date = tdate if tdate is not None else last_date
            if row_date is None:
                continue

            # NEW: use line vertical bounds for precise ROI cropping
            line_y0 = line.get("y0")
            line_y1 = line.get("y1")

            # Amounts (require decimals to avoid giant integer false-positives)
            debit = _amount_with_numeric_fallback(
                words, windows["withdrawn_window"], page_img_path, line_y0, line_y1,
                pad=WINDOW_PAD, merge_gap_px=MERGE_GAP, require_decimal=True
            )
            credit = _amount_with_numeric_fallback(
                words, windows["paidin_window"], page_img_path, line_y0, line_y1,
                pad=WINDOW_PAD, merge_gap_px=MERGE_GAP, require_decimal=True
            )
            bal = _amount_with_numeric_fallback(
                words, windows["balance_window"], page_img_path, line_y0, line_y1,
                pad=WINDOW_PAD, merge_gap_px=MERGE_GAP, require_decimal=True
            )

            # Must have debit or credit to count as a transaction
            if (debit is None or debit == 0.0) and (credit is None or credit == 0.0):
                continue

            if bal is not None:
                last_balance_seen = float(bal)

            raw_compact_date = _find_compact_date_token(ltxt)
            clean_desc = _description_from_words(words, windows, raw_compact_date)
            if not clean_desc:
                first_amt_left = min(float(windows["withdrawn_window"][0]), float(windows["paidin_window"][0]))
                kept = [_norm(w.get("text", "")) for w in words if float(w.get("left", 0)) < first_amt_left - 4]
                clean_desc = re.sub(r"\s{2,}", " ", " ".join(kept)).strip(" -|")

            tx_type = "credit" if (credit or 0.0) > 0 else "debit"
            amt_val = float(credit if (credit or 0.0) > 0 else (debit or 0.0))

            all_transactions.append({
                "seq": seq,
                "transaction_date": row_date,
                "transaction_type": tx_type,
                "description": clean_desc,
                "amount": amt_val,
                "signed_amount": (amt_val if tx_type == "credit" else -amt_val),
            })

            seq += 1

            debug_rows.append({
                "page": page_num,
                "date": row_date,
                "line": ltxt,
                "debit": float(debit) if debit is not None else None,
                "credit": float(credit) if credit is not None else None,
                "balance": float(bal) if bal is not None else None,
            })


    return all_transactions, debug_rows, last_balance_seen


# ---------------------------------
# Currency aggregation
# ---------------------------------
def _group_all_to_single_currency(transactions: List[dict], currency: str) -> Dict[str, List[dict]]:
    return {currency: sorted(transactions, key=lambda t: t.get("seq", 0))}

def _build_currency_sections_from_rows(
    buckets: Dict[str, List[dict]],
    opening_by_cur: Dict[str, Optional[float]],
    closing_by_cur: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cur, txs in sorted(buckets.items()):
        money_in  = round(sum(float(t["amount"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out = round(sum(float(t["amount"]) for t in txs if t.get("transaction_type") == "debit"), 2)

        open_from = opening_by_cur.get(cur)
        close_stmt = closing_by_cur.get(cur)

        close_calc = None
        if open_from is not None:
            close_calc = round(float(open_from) + money_in - money_out, 2)

        out[cur] = {
            "balances": {
                "opening_balance": {
                    "summary_table": None,
                    "transactions_table": None if open_from is None else float(open_from),
                },
                "money_in_total": {
                    "summary_table": None,
                    "transactions_table": float(money_in),
                },
                "money_out_total": {
                    "summary_table": None,
                    "transactions_table": float(money_out),
                },
                "closing_balance": {
                    "summary_table": None,
                    "transactions_table": None if close_stmt is None else float(close_stmt),
                    "calculated": None if close_calc is None else float(close_calc),
                },
            },
            "transactions": txs,
        }
    return out


# ---------------------------------
# IBAN extraction
# ---------------------------------
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


# ---------------------------------
# Public entrypoint
# ---------------------------------
def parse_statement(raw_ocr, client: str = "Unknown", account_type: str = "Unknown", debug: bool = False) -> dict:
    """
    Returns a single 'statement node' (no top-level 'client'), aligned with AIB/BOI/N26.
    """
    pages = raw_ocr.get("pages", []) or []

    iban = extract_iban(pages)
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    # Opening/Closing via explicit labelled rows (with robust rescue)
    explicit_open, explicit_close = extract_opening_closing_ptsb(pages, debug=debug)

    # Transactions (and remember last Balance for fallback closing)
    transactions, debug_rows, last_balance_seen = parse_transactions_ptsb(pages, iban=iban)

    # Group to single-currency buckets
    buckets = _group_all_to_single_currency(transactions, currency)

    # Opening/Closing by currency
    opening_by_cur = {currency: explicit_open}
    closing_by_cur = {currency: explicit_close if explicit_close is not None else last_balance_seen}

    # Currency sections (strict schema)
    currencies = _build_currency_sections_from_rows(buckets, opening_by_cur, closing_by_cur)

    # Statement dates
    if transactions:
        all_dates = [t.get("transaction_date") for t in transactions if t.get("transaction_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = end_date = None

    # Optional debug CSV
    if debug and debug_rows:
        try:
            import os, csv
            os.makedirs("results", exist_ok=True)
            dbg_csv = os.path.join("results", "ptsb_debug_rows.csv")
            with open(dbg_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["page","date","line","debit","credit","balance"])
                w.writeheader()
                w.writerows(debug_rows)
            print(f"🧪 PTSB debug rows saved to {dbg_csv}")
        except Exception as e:
            print(f"⚠️ Failed to write ptsb_debug_rows.csv: {e}")

    # Lightweight statement_id (same pattern as AIB/BOI/N26)
    sid_basis = f"{raw_ocr.get('file_name') or ''}|{start_date or ''}|{end_date or ''}"
    statement_id = hashlib.sha1(sid_basis.encode("utf-8")).hexdigest()[:12] if sid_basis.strip("|") else None

    return {
        "statement_id": statement_id,
        "file_name": raw_ocr.get("file_name"),
        "institution": "PTSB",
        "account_type": account_type,
        "account_holder": None,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
