# n26_current.py
# Parse N26 statements into the same JSON shape used by Revolut/BOI.
# - Find the true header row ("Booking", "Date", "Amount") to anchor columns.
# - Build rows via Y-clusters; keep only clusters that contain an amount within the Amount window.
# - Date = topmost date token within the date window on the row.
# - Description = all tokens on the same baseline strictly left of the Booking Date column (seed + baseline grow).
# - Dedupe rows by (page, y-bin, date, amount).

from __future__ import annotations
import re
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import pandas as pd

from utils import parse_currency, parse_date

# ---------------- regexes ----------------
DATE_RE = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b")
# Keep the regex permissive, but we’ll require either a sign or a € in-window (to avoid dates)
AMT_RE = re.compile(r"(?P<sign>[+\-−])?\s*(?P<body>\d{1,3}(?:[.\s\u00A0\u2009\u202F]\d{3})*[.,]\d{2})\s*(?:€)?")

# ---------------- small utils ----------------
def _norm(s: str) -> str:
    if not s: return ""
    return (str(s)
            .replace("\u00A0", " ")
            .replace("\u2009", " ")
            .replace("\u202F", " ")
            .replace("−", "-")
            .strip())

def _page_words_to_df(page: dict) -> pd.DataFrame:
    rows = []
    for ln in page.get("lines", []) or []:
        for w in ln.get("words", []) or []:
            rows.append(w)
    if not rows:
        return pd.DataFrame(columns=["text", "left", "top", "width", "height",
                                     "right", "bottom", "cy",
                                     "block_num", "par_num", "line_num"])
    df = pd.DataFrame(rows).copy()
    for c in ("left", "top", "width", "height"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else: df[c] = 0
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    df["cy"] = df["top"] + df["height"] * 0.5
    df["text"] = df["text"].astype(str)
    # tesseract IDs (optional)
    for c in ("block_num", "par_num", "line_num"):
        if c not in df.columns: df[c] = -1
    return df

def _med_height(df: pd.DataFrame) -> float:
    return float(df["height"].median()) if "height" in df and not df["height"].empty else 18.0

def _row_key(anchor_y: float, med_h: float) -> int:
    # Wider bin to collapse duplicates
    ybin_h = max(10.0, med_h * 0.9)
    return int(round(float(anchor_y) / ybin_h))

# ---------------- header detection ----------------
def _find_header_columns_by_words(pages: List[dict]) -> Optional[Dict[str, int]]:
    """
    Look for the single header line that contains 'Booking' and 'Date' and 'Amount'.
    Returns dict with date/amount windows and the hard left edge of the date column.
    """
    for page in pages[:3]:
        df = _page_words_to_df(page)
        if df.empty: continue

        tok = df.assign(t=df["text"].str.strip().str.lower())
        # "booking" and "date" can be separate tokens
        has_booking = tok["t"].str.contains(r"^booking$", regex=True)
        has_date    = tok["t"].str.contains(r"^date$", regex=True)
        has_amount  = tok["t"].str.contains(r"^amount$", regex=True)

        # find any line that contains all three words across its words set
        for ln_id, ln in df.groupby(["block_num", "par_num", "line_num"]):
            words = ln["text"].str.strip().str.lower().tolist()
            if not words:
                continue
            s = " ".join(words)
            if ("booking" in s and "date" in s) and ("amount" in s):
                # derive x-positions
                amt_words = ln[ln["text"].str.strip().str.lower().eq("amount")]
                if amt_words.empty:
                    continue
                amount_left = int(amt_words["left"].median())
                amount_right = int((amt_words["left"] + amt_words["width"]).median())

                # date can be "booking" and "date" separated; take the span from first of them to the end of "date"
                date_tokens = ln[ln["text"].str.strip().str.lower().isin(["booking", "date"])]
                if date_tokens.empty:
                    continue
                date_left = int(date_tokens["left"].min())
                date_right = int((date_tokens["left"] + date_tokens["width"]).max())

                pad = 36
                desc_max_right = date_left - 40

                return dict(
                    date_left=date_left - pad,
                    date_right=date_right + pad,
                    amount_left=amount_left - 80,
                    amount_right=amount_right + 140,
                    desc_max_right=desc_max_right,
                    date_left_hard=date_left,
                )
    return None

# ---------------- y-clusters ----------------
def _yclusters(df: pd.DataFrame, gap: Optional[float] = None) -> List[pd.DataFrame]:
    """Group tokens into coarse horizontal rows by center-Y."""
    if df.empty: return []
    med_h = _med_height(df)
    if gap is None:
        gap = max(6.0, med_h * 0.45)  # tight enough not to merge adjacent rows
    df2 = df.sort_values("cy").reset_index(drop=True)
    clusters: List[pd.DataFrame] = []
    start = 0
    for i in range(1, len(df2)):
        if df2.loc[i, "cy"] - df2.loc[i - 1, "cy"] > gap:
            clusters.append(df2.iloc[start:i].sort_values("left"))
            start = i
    clusters.append(df2.iloc[start:].sort_values("left"))
    return clusters

# ---------------- field extractors ----------------
def _extract_topmost_date_in_window(row_df: pd.DataFrame, left: int, right: int) -> tuple[Optional[str], Optional[int]]:
    """Return (date_text 'DD.MM.YYYY', anchor_y from the chosen date token cy)."""
    if row_df.empty: return None, None
    m = (row_df["right"] >= left) & (row_df["left"] <= right)
    cand = row_df.loc[m, ["text", "top", "height", "cy"]].copy()
    if cand.empty: return None, None

    toks: List[Tuple[str, int, float]] = []
    for _, r in cand.iterrows():
        for t in str(r["text"]).split():
            mm = DATE_RE.search(t)
            if mm:
                toks.append((mm.group(0), int(r["top"]), float(r.get("cy", r["top"] + r["height"] * 0.5))))

    if toks:
        tmin, top, cy = min(toks, key=lambda x: x[1])
        m2 = DATE_RE.search(tmin)
        d, mth, y = m2.groups()
        if len(y) == 2: y = "20" + y
        return f"{d.zfill(2)}.{mth.zfill(2)}.{y}", int(round(cy))

    # fallback: scan concatenated window text
    txt = _norm(" ".join(cand["text"].tolist())).replace(" ", "")
    m3 = DATE_RE.search(txt)
    if not m3:
        return None, None
    d, mth, y = m3.groups()
    if len(y) == 2: y = "20" + y
    anchor_y = int(round(float(cand["cy"].min())))
    return f"{d.zfill(2)}.{mth.zfill(2)}.{y}", anchor_y

def _extract_amount_text(row_df: pd.DataFrame, left: int, right: int) -> Optional[str]:
    """Extract the visible amount text in the amount window. Guard against dates."""
    if row_df.empty: return None
    in_win = row_df[(row_df["right"] >= left) & (row_df["left"] <= right)]
    if in_win.empty: return None
    txt = _norm(" ".join(in_win["text"].tolist()))
    m = AMT_RE.search(txt)
    if not m:
        return None
    # extra guard: require either a sign or a euro symbol in the window text
    raw_win = "".join(in_win["text"].astype(str).tolist())
    if ("€" not in raw_win) and (m.group("sign") not in ("+","-","−")):
        return None
    sign = m.group("sign") or ""
    sign = "-" if sign in ("-","−") else ("+" if sign == "+" else "")
    body = m.group("body").replace(" ", "").replace("\u00A0", "").replace("\u2009", "").replace("\u202F", "")
    if "," not in body and "." in body:
        i = body.rfind(".")
        body = body[:i] + "," + body[i + 1:]
    return f"{sign}{body}€"

# ---------------- description (seed + baseline) ----------------
def _left_of_date_lines(page_df: pd.DataFrame, date_left_hard: int) -> list[pd.DataFrame]:
    if page_df is None or page_df.empty:
        return []
    left_cut = int(date_left_hard) - 2
    left = page_df[page_df["left"] < left_cut].copy()
    if left.empty:
        return []
    left = left.sort_values("cy").reset_index(drop=True)
    med_h = _med_height(left)
    gap = max(6.0, med_h * 0.55)  # roughly one visual line per group
    lines = []
    start = 0
    for i in range(1, len(left)):
        if left.loc[i, "cy"] - left.loc[i - 1, "cy"] > gap:
            lines.append(left.iloc[start:i].sort_values("left"))
            start = i
    lines.append(left.iloc[start:].sort_values("left"))
    return lines

def _choose_desc_line(lines: list[pd.DataFrame], anchor_y: int) -> Optional[pd.DataFrame]:
    if not lines:
        return None
    scored = []
    for ln in lines:
        if ln.empty: continue
        txt = " ".join(_norm(t) for t in ln["text"].astype(str)).strip()
        if not txt: continue
        low = txt.lower()
        if low.startswith("value date") or "mastercard" in low:
            continue
        cy = float(ln["cy"].median())
        width = int(ln["right"].max() - ln["left"].min())
        scored.append({"cy": cy, "width": width, "df": ln})
    if not scored:
        return None
    # prefer lines above/same as anchor, nearest first; tie-break: widest
    above = [s for s in scored if s["cy"] <= float(anchor_y) + 1.0]
    if above:
        above.sort(key=lambda s: (float(anchor_y) - s["cy"], -s["width"]))
        return above[0]["df"]
    scored.sort(key=lambda s: (abs(s["cy"] - float(anchor_y)), -s["width"]))
    return scored[0]["df"]

def _description_from_anchor(page_df: pd.DataFrame, date_left_hard: int, anchor_y: int) -> Optional[str]:
    best = _choose_desc_line(_left_of_date_lines(page_df, date_left_hard), anchor_y)
    if best is None or best.empty:
        return None
    text = " ".join(_norm(t) for t in best.sort_values("left")["text"].astype(str)).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text or None

# ---------------- parsing core ----------------
def _infer_currency_from_iban(iban: Optional[str], fallback: Optional[str] = "EUR") -> str:
    if not iban:
        return fallback or "EUR"
    u = iban.upper()
    if u.startswith("GB"): return "GBP"
    # Most N26 are DE/IE → EUR
    return "EUR"

def _to_iso_date(s: str) -> Optional[str]:
    """Accept 'DD.MM.YYYY' and return ISO 'YYYY-MM-DD'; otherwise try parse_date()."""
    s = s.strip()
    m = DATE_RE.search(s)
    if not m:
        return parse_date(s)  # let utils handle other formats
    d, mth, y = m.groups()
    if len(y) == 2: y = "20" + y
    try:
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
    except Exception:
        return parse_date(s)

def _amount_to_number(a: str) -> Optional[float]:
    if not a:
        return None
    # reuse utils.parse_currency (robust to €/comma/period)
    v = parse_currency(a, strip_currency=False)
    try:
        return None if v is None else float(v)
    except Exception:
        return None

def parse_transactions(pages: List[dict], iban: Optional[str] = None) -> List[dict]:
    """
    Build flat list of N26 transactions using page-level words + Y-clusters,
    anchored by the true header row.
    """
    currency = _infer_currency_from_iban(iban, "EUR")
    out: List[dict] = []
    seq = 0

    # 1) header → column windows
    cols = _find_header_columns_by_words(pages)
    if not cols:
        return out  # fail-safe: no header -> nothing

    seen: set[tuple] = set()  # dedupe key
    for page_idx, page in enumerate(pages):
        df_page = _page_words_to_df(page)
        if df_page.empty:
            continue

        med_h_page = _med_height(df_page)

        for row in _yclusters(df_page):
            amount_txt = _extract_amount_text(row, cols["amount_left"], cols["amount_right"])
            if not amount_txt:
                continue

            date_txt, anchor_y = _extract_topmost_date_in_window(row, cols["date_left"], cols["date_right"])
            if not date_txt or anchor_y is None:
                continue

            # Description from the page, anchored at the booking-date row
            desc = _description_from_anchor(df_page, cols["date_left_hard"], anchor_y) or ""

            # Dedup by coarse rowbin + (date, amount)
            rowbin = _row_key(anchor_y, med_h_page)
            key = (page_idx, rowbin, date_txt, amount_txt)
            if key in seen:
                continue
            seen.add(key)

            iso_date = _to_iso_date(date_txt)
            val = _amount_to_number(amount_txt)
            if val is None:
                continue

            out.append({
                "seq": seq,
                "transactions_date": iso_date,
                "transaction_type": "credit" if float(val) > 0 else "debit",
                "description": desc,
                "amount": {
                    "value": abs(float(val)),  # keep value positive; type encodes direction
                    "currency": currency,
                },
                # N26 PDFs don’t carry a running balance per row
                "balance_after_statement": None,
                "balance_after_calculated": None,
            })
            seq += 1

    # preserve statement order
    out.sort(key=lambda t: t.get("seq", 0))
    return out

# ---------------- currency sections (N26: no per-page totals) ----------------
def _group_by_currency(transactions: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in transactions:
        cur = (t.get("amount") or {}).get("currency")
        if cur:
            buckets[cur].append(t)
    for cur in buckets:
        buckets[cur].sort(key=lambda t: t.get("seq", 0))
    return buckets

def _build_currency_sections_from_rows(buckets: Dict[str, List[dict]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cur, txs in sorted(buckets.items()):
        money_in  = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit"), 2)
        # N26 statements don’t include balance columns; keep opening/closing unknown
        out[cur] = {
            "opening_balance": None,
            "money_in_total":  {"value": money_in,  "currency": cur},
            "money_out_total": {"value": money_out, "currency": cur},
            "closing_balance_statement": None,
            "closing_balance_calculated": None,
            "transactions": txs,
        }
    return out

# ---------------- public entrypoint ----------------
def parse_statement(raw_ocr, client: str = "Unknown", account_type: str = "Unknown"):
    """
    Public API to align with Revolut/BOI parse_statement signature/shape.
    """
    pages = raw_ocr.get("pages", []) or []
    # Flat text for IBAN/BIC extraction
    full_text = "\n".join(
        "\n".join((ln.get("line_text") or "") for ln in pg.get("lines", []))
        for pg in pages
    )

    # IBAN / BIC (simple)
    m_iban = re.search(r"\bIBAN[:\s]+([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b", full_text, re.IGNORECASE)
    iban = m_iban.group(1).replace(" ", "") if m_iban else None
    m_bic = re.search(r"\bBIC[:\s]+([A-Z0-9]{8,11})\b", full_text, re.IGNORECASE)
    bic = m_bic.group(1) if m_bic else None

    # 1) transactions
    transactions = parse_transactions(pages, iban=iban)

    # 2) group per currency and compute simple totals from rows
    buckets = _group_by_currency(transactions)
    currencies = _build_currency_sections_from_rows(buckets)

    # 3) statement dates from transactions
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,     # add later if you choose to parse it
        "institution": "N26",
        "account_type": account_type,
        "iban": iban,
        "bic": bic,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
