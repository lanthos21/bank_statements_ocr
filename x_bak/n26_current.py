# n26_current.py
# N26 parser with:
#   • robust row extraction (full Description),
#   • multi-page, multi-pocket summary table aggregation,
#   • overlay of opening/closing + incoming/outgoing from the tables into the standard JSON,
#   • rows remain independent (validator cross-checks totals vs tx sums).

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from utils import parse_currency, parse_date


# ---------------- regexes ----------------
DATE_RE = re.compile(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b")

# Summary labels (we match via token set, but keep simple line-based RE backups)
RE_PREV = re.compile(r"^\s*previous\s+balance\b", re.I)
RE_OUT  = re.compile(r"^\s*outgoing\s+transactions\b", re.I)
RE_IN   = re.compile(r"^\s*incoming\s+transactions\b", re.I)
RE_NEW  = re.compile(r"^\s*(your\s+new\s+balance|new\s+balance)\b", re.I)

# Monetary amount (tolerant to thin/nb spaces, 1k separators, sign)
AMT_RE  = re.compile(r"(?P<sign>[+\-−])?\s*(?P<body>\d{1,3}(?:[.\s\u00A0\u2009\u202F]\d{3})*[.,]\d{2})\s*(?:€)?")


# ---------------- small utils ----------------
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


def _page_words_to_df(page: dict) -> pd.DataFrame:
    rows = []
    for ln in page.get("lines", []) or []:
        for w in ln.get("words", []) or []:
            rows.append(w)
    if not rows:
        return pd.DataFrame(
            columns=[
                "text", "left", "top", "width", "height",
                "right", "bottom", "cy", "block_num", "par_num", "line_num"
            ]
        )
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


def _row_key(anchor_y: float, med_h: float) -> int:
    ybin_h = max(10.0, med_h * 0.9)
    return int(round(float(anchor_y) / ybin_h))


# ---------------- header detection ----------------
def _find_header_columns_by_words(pages: List[dict]) -> Optional[Dict[str, int]]:
    """
    Find the 'Booking Date' + 'Amount' header; returns column windows and a hard left date cut.
    """
    for page in pages[:3]:
        df = _page_words_to_df(page)
        if df.empty:
            continue

        for _, ln in df.groupby(["block_num", "par_num", "line_num"]):
            words = ln["text"].str.strip().str.lower().tolist()
            if not words:
                continue
            s = " ".join(words)
            if ("booking" in s and "date" in s) and ("amount" in s):
                amt_words = ln[ln["text"].str.strip().str.lower().eq("amount")]
                if amt_words.empty:
                    continue
                amount_left  = int(amt_words["left"].median())
                amount_right = int((amt_words["left"] + amt_words["width"]).median())

                date_tokens = ln[ln["text"].str.strip().str.lower().isin(["booking", "date"])]
                if date_tokens.empty:
                    continue
                date_left  = int(date_tokens["left"].min())
                date_right = int((date_tokens["left"] + date_tokens["width"]).max())

                pad = 36
                desc_max_right = date_left - 40
                return dict(
                    date_left=date_left - pad,
                    date_right=date_right + pad,
                    amount_left=amount_left - 80,
                    amount_right=amount_right + 140,
                    desc_max_right=desc_max_right,
                    date_left_hard=date_left
                )
    return None


# ---------------- y-clusters ----------------
def _yclusters(df: pd.DataFrame, gap: Optional[float] = None) -> List[pd.DataFrame]:
    if df.empty:
        return []
    med_h = _med_height(df)
    gap = max(6.0, med_h * 0.45) if gap is None else gap
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
    if row_df.empty:
        return None, None
    m = (row_df["right"] >= left) & (row_df["left"] <= right)
    cand = row_df.loc[m, ["text", "top", "height", "cy"]].copy()
    if cand.empty:
        return None, None

    toks: List[Tuple[str, int, float]] = []
    for _, r in cand.iterrows():
        for t in str(r["text"]).split():
            mm = DATE_RE.search(t)
            if mm:
                toks.append((mm.group(0), int(r["top"]), float(r.get("cy", r["top"] + r["height"] * 0.5))))
    if toks:
        tmin, _top, cy = min(toks, key=lambda x: x[1])
        d, mth, y = DATE_RE.search(tmin).groups()
        if len(y) == 2:
            y = "20" + y
        return f"{int(d):02d}.{int(mth):02d}.{int(y):04d}", int(round(cy))

    txt = _norm(" ".join(cand["text"].tolist())).replace(" ", "")
    m3 = DATE_RE.search(txt)
    if not m3:
        return None, None
    d, mth, y = m3.groups()
    if len(y) == 2:
        y = "20" + y
    anchor_y = int(round(float(cand["cy"].min())))
    return f"{int(d):02d}.{int(mth):02d}.{int(y):04d}", anchor_y


def _extract_amount_text(row_df: pd.DataFrame, left: int, right: int) -> Optional[str]:
    if row_df.empty:
        return None
    in_win = row_df[(row_df["right"] >= left) & (row_df["left"] <= right)]
    if in_win.empty:
        return None
    txt = _norm(" ".join(in_win["text"].tolist()))
    m = AMT_RE.search(txt)
    if not m:
        return None

    # guard: require either explicit sign or '€' present among tokens (cuts false positives)
    raw_win = "".join(in_win["text"].astype(str).tolist())
    if ("€" not in raw_win) and (m.group("sign") not in ("+", "-", "−")):
        return None

    sign = m.group("sign") or ""
    sign = "-" if sign in ("-", "−") else ("+" if sign == "+" else "")
    body = (
        m.group("body")
        .replace(" ", "")
        .replace("\u00A0", "")
        .replace("\u2009", "")
        .replace("\u202F", "")
    )
    if "," not in body and "." in body:
        i = body.rfind(".")
        body = body[:i] + "," + body[i + 1 :]
    return f"{sign}{body}€"


# ---------------- description via anchor/baseline ----------------
def _left_of_date_lines(page_df: pd.DataFrame, date_left_hard: int) -> list[pd.DataFrame]:
    if page_df is None or page_df.empty:
        return []
    left_cut = int(date_left_hard) - 2
    left = page_df[page_df["left"] < left_cut].copy()
    if left.empty:
        return []
    left = left.sort_values("cy").reset_index(drop=True)
    med_h = _med_height(left)
    gap = max(6.0, med_h * 0.55)
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
        if ln.empty:
            continue
        txt = " ".join(_norm(t) for t in ln["text"].astype(str)).strip()
        if not txt:
            continue
        low = txt.lower()
        if low.startswith("value date") or "mastercard" in low:  # ignore furniture
            continue
        cy = float(ln["cy"].median())
        width = int(ln["right"].max() - ln["left"].min())
        scored.append({"cy": cy, "width": width, "df": ln})
    if not scored:
        return None
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


# ---------------- summary helpers (token-band approach) ----------------
def _alpha_tokens(df: pd.DataFrame) -> list[str]:
    """Normalize tokens to simple lowercase a-z (e.g., 'Previous,' -> 'previous')."""
    return [re.sub(r"[^a-z]+", "", (t or "").lower()) for t in df["text"].astype(str)]


def _label_kind_from_tokens(tokens: list[str]) -> Optional[str]:
    toks = {t for t in tokens if t}
    def has(*ws): return all(w in toks for w in ws)
    if has("previous", "balance"): return "previous"
    if has("outgoing", "transactions"): return "out"
    if has("incoming", "transactions"): return "in"
    if has("new", "balance") or has("your", "new", "balance"): return "new"
    return None


def _rightmost_amount(words: list[dict]) -> float | None:
    """
    Return the rightmost amount (as float) in a token list, tolerant to '€' being a separate token.
    """
    if not words:
        return None
    cands = []
    n = len(words)
    for i, w in enumerate(words):
        t = _norm(w.get("text", "") or "")
        if not t:
            continue
        left = int(w.get("left", 0))
        width = int(w.get("width", 0))
        xr = left + width

        combined = t
        if "€" not in t and i + 1 < n:
            nxt = _norm(words[i + 1].get("text", "") or "")
            if "€" in nxt:
                combined = t + nxt
                xr = int(words[i + 1].get("left", 0)) + int(words[i + 1].get("width", 0))

        m = AMT_RE.search(combined)
        if not m:
            continue

        # Filter out date tails like ".2024" accidentally glued after the amount
        if re.match(r"\.\s*\d{4}\b", combined[m.end():]):
            continue

        txt = (m.group("sign") or "") + m.group("body")
        txt = txt.replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ").replace(" ", "")
        if "," in txt:
            txt = txt.replace(".", "").replace(",", ".")
        try:
            val = float(txt)
        except ValueError:
            continue
        cands.append((xr, round(val, 2)))
    if not cands:
        return None
    cands.sort(key=lambda p: p[0])
    return cands[-1][1]


def _rightmost_amount_near_y(
    df: pd.DataFrame,
    anchor_y: float,
    x_min: Optional[int] = None,
    span_mult: float = 1.1,
) -> Optional[float]:
    """
    Find the rightmost amount near a vertical baseline (anchor_y).
    Optionally ignore tokens left of x_min (helps when labels are at far left).
    """
    if df is None or df.empty:
        return None

    med_h = _med_height(df)

    def pick(win_mult: float, require_right_of: Optional[int]) -> Optional[float]:
        y0 = float(anchor_y) - win_mult * med_h
        y1 = float(anchor_y) + win_mult * med_h
        band = df[df["cy"].between(y0, y1)].sort_values("left")
        if require_right_of is not None:
            band = band[band["left"] >= int(require_right_of) - 6]
        if band.empty:
            return None
        words = band[["text", "left", "width"]].to_dict("records")
        return _rightmost_amount(words)

    amt = pick(span_mult, x_min)
    if amt is None:
        amt = pick(span_mult * 1.6, x_min)
    if amt is None and x_min is not None:
        amt = pick(span_mult * 1.6, None)
    return amt


def collect_n26_summary_tables_all(
    pages: list[dict],
    infer_page_currency,
    debug: bool = False
) -> dict[str, dict]:
    """
    Find and SUM every N26 summary table across all pages/pockets.
    Returns {CUR: {"previous": x, "out": y, "in": z, "new": w}} (all 2dp).
    Works even on “Spaces” pages where OCR jumbles words across columns.
    """
    totals: dict[str, dict] = {}
    prev_cur = None

    for pidx, page in enumerate(pages):
        cur = infer_page_currency(page, prev_cur) or "EUR"
        prev_cur = cur
        if cur not in totals:
            totals[cur] = {"previous": 0.0, "out": 0.0, "in": 0.0, "new": 0.0}

        df = _page_words_to_df(page)
        if df.empty:
            if debug:
                print(f"[SUMM p{pidx+1}] no rows")
            continue

        med_h = _med_height(df)
        bin_h = max(12.0, med_h * 1.20)
        df = df.copy()
        df["ybin"] = (df["cy"] / bin_h).round().astype(int)

        page_vals = {"previous": 0.0, "out": 0.0, "in": 0.0, "new": 0.0}
        seen_bins: set[tuple[str, int]] = set()
        found_any = False

        for yb, band in df.groupby("ybin"):
            band_sorted = band.sort_values("left")
            tokens = _alpha_tokens(band_sorted)
            kind = _label_kind_from_tokens(tokens)
            if not kind:
                continue

            key = (kind, int(yb))
            if key in seen_bins:
                continue
            seen_bins.add(key)

            anchor_y = float(band_sorted["cy"].median())
            x_min = int(band_sorted["left"].min())

            amt = _rightmost_amount_near_y(df, anchor_y, x_min=x_min)
            if amt is None:
                # last resort: only this band's words
                amt = _rightmost_amount(band_sorted[["text", "left", "width"]].to_dict("records"))
            if amt is None:
                continue

            found_any = True
            val = round(float(amt), 2)
            if kind == "previous":
                page_vals["previous"] += val
            elif kind == "new":
                page_vals["new"] += val
            elif kind == "out":
                page_vals["out"] += abs(val)
            elif kind == "in":
                page_vals["in"] += abs(val)

        if found_any:
            # accumulate per page (summing pockets on that page)
            totals[cur]["previous"] = round(totals[cur]["previous"] + page_vals["previous"], 2)
            totals[cur]["new"]      = round(totals[cur]["new"]      + page_vals["new"],      2)
            totals[cur]["out"]      = round(totals[cur]["out"]      + page_vals["out"],      2)
            totals[cur]["in"]       = round(totals[cur]["in"]       + page_vals["in"],       2)
            if debug:
                print(f"[SUMM p{pidx+1}] page_vals={page_vals} → totals[{cur}]={totals[cur]}")
        else:
            if debug:
                print(f"[SUMM p{pidx+1}] no labeled bands")

    return totals


# ---------------- helpers ----------------
def _infer_currency_from_iban(iban: Optional[str], fallback: Optional[str] = "EUR") -> str:
    if not iban:
        return fallback or "EUR"
    u = iban.upper()
    if u.startswith("GB"):
        return "GBP"
    return "EUR"


def _to_iso_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    m = DATE_RE.search(s)
    if not m:
        return parse_date(s)
    d, mth, y = m.groups()
    if len(y) == 2:
        y = "20" + y
    try:
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"
    except Exception:
        return parse_date(s)


def _amount_to_number(a: str) -> Optional[float]:
    if not a:
        return None
    v = parse_currency(a, strip_currency=False)
    try:
        return None if v is None else float(v)
    except Exception:
        return None


# ---------------- transactions ----------------
def parse_transactions(pages: List[dict], iban: Optional[str] = None) -> List[dict]:
    currency = _infer_currency_from_iban(iban, "EUR")
    out: List[dict] = []
    seq = 0

    cols = _find_header_columns_by_words(pages)
    if not cols:
        return out

    seen: set[tuple] = set()

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

            desc = _description_from_anchor(df_page, cols["date_left_hard"], anchor_y) or ""

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
                    "value": abs(float(val)),  # positive magnitude
                    "currency": currency,
                },
                "balance_after_statement": None,
                "balance_after_calculated": None,
            })
            seq += 1

    out.sort(key=lambda t: t.get("seq", 0))
    return out


# ---------------- currency sections from rows ----------------
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
        money_in_rows  = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out_rows = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit"), 2)
        out[cur] = {
            "opening_balance": None,  # will overlay from tables if found
            "money_in_total":  {"value": money_in_rows,  "currency": cur},   # may be overwritten by tables
            "money_out_total": {"value": money_out_rows, "currency": cur},   # may be overwritten by tables
            "closing_balance_statement": None,                                # will overlay from tables if found
            "closing_balance_calculated": None,
            "transactions": txs,
        }
    return out


# ---------------- public entrypoint ----------------
def parse_statement(raw_ocr, client: str = "Unknown", account_type: str = "Unknown"):
    """
    Build the unified JSON:
      currencies[CUR] = opening_balance / money_in_total / money_out_total / closing_balance_statement
      are filled from the N26 summary tables (summed across pockets & pages),
      while transactions list is parsed from rows (used by your validator to cross-check).
    """
    pages = raw_ocr.get("pages", []) or []

    # Flat text for IBAN/BIC
    full_text = "\n".join("\n".join((ln.get("line_text") or "") for ln in pg.get("lines", [])) for pg in pages)
    m_iban = re.search(r"\bIBAN[:\s]+([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b", full_text, re.IGNORECASE)
    iban = m_iban.group(1).replace(" ", "") if m_iban else None
    m_bic = re.search(r"\bBIC[:\s]+([A-Z0-9]{8,11})\b", full_text, re.IGNORECASE)
    bic = m_bic.group(1) if m_bic else None

    def _infer(page, prev):
        cur = page.get("currency")
        if isinstance(cur, str) and len(cur) == 3:
            return cur.upper()
        if iban:
            u = iban.upper()
            if u.startswith("GB"):
                return "GBP"
            if u.startswith("IE"):
                return "EUR"
        return prev or _infer_currency_from_iban(iban, "EUR")

    # 1) parse transactions (rows)
    transactions = parse_transactions(pages, iban=iban)

    # 2) group rows by currency and build currency nodes
    buckets = _group_by_currency(transactions)
    currencies = _build_currency_sections_from_rows(buckets)

    # 3) collect + SUM all summary tables across all pages/pockets
    summ = collect_n26_summary_tables_all(pages, _infer, debug=False)  # -> { "EUR": {previous,out,in,new}, ... }

    # 4) overlay summary values into the standard fields (so validator treats it as “Revolut-style”)
    for cur, s in summ.items():
        if cur not in currencies:
            # If there are tables but no transactions (edge case), still emit a currency node
            currencies[cur] = {
                "opening_balance": None,
                "money_in_total": None,
                "money_out_total": None,
                "closing_balance_statement": None,
                "closing_balance_calculated": None,
                "transactions": [],
            }

        prev_b = s.get("previous")
        new_b  = s.get("new")
        in_s   = abs(s.get("in", 0.0))   # ensure positive magnitudes
        out_s  = abs(s.get("out", 0.0))

        currencies[cur]["opening_balance"] = None if prev_b is None else {"value": float(prev_b), "currency": cur}
        currencies[cur]["closing_balance_statement"] = None if new_b is None else {"value": float(new_b), "currency": cur}
        currencies[cur]["money_in_total"]  = {"value": float(round(in_s, 2)),  "currency": cur}
        currencies[cur]["money_out_total"] = {"value": float(round(out_s, 2)), "currency": cur}

    # 5) statement dates from transactions
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,
        "institution": "N26",
        "account_type": account_type,
        "iban": iban,
        "bic": bic,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
