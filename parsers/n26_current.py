# n26_current.py — single-pass OCR parser for N26 using header X-positions
# Works with your OCR module output: raw_ocr["pages"][i]["lines"][j]["words"][k] dicts

import re
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple

import pandas as pd
from utils import parse_currency, parse_date

DEBUG = True

# ---------- Patterns ----------
# replace your patterns with these
DATE_DOT_REGEX = re.compile(r"\b\d{1,2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{2,4}\b")

# Accepts "€ 1.234,56", "1.234,56 €", "−1 234,56", "1 234,56", etc.
AMOUNT_CHUNK_REGEX = re.compile(
    r"(?:€\s*)?[-+−]?\s*\d{1,3}(?:[.\s \u00A0\u2009\u202F]\d{3})*[.,]\d{2}\s*(?:€)?"
)


DATE_ANYWHERE_RE = re.compile(
    r"\b(\d{1,2}\s*[\.,/-]\s*\d{1,2}\s*[\.,/-]\s*\d{2,4})\b"
)

# Noise / furniture
FOOTER_RE      = re.compile(r"\buntil\s+\d+\s*/\s*\d+\b", re.IGNORECASE)
VALUE_DATE_RE  = re.compile(r"\bvalue\s+date\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}\b", re.IGNORECASE)
SUMMARY_ROW_RE = re.compile(r"^\s*(incoming|outgoing|previous)\s*$", re.IGNORECASE)

# Column tolerance
DATE_WIN   = 40
AMOUNT_WIN = 140

# ---------- OCR helpers ----------
def _norm(s: str) -> str:
    if not s:
        return ""
    s = (s.replace("\u00A0", " ")
           .replace("\u2009", " ")
           .replace("\u202F", " ")
           .replace("−", "-").replace("–", "-").replace("—", "-")
           .replace("·", ".").replace("•", "."))
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def _extract_date_from_df_col(df: pd.DataFrame, date_right: int) -> Optional[str]:
    if df.empty or date_right < 0:
        return None
    mask = (df["right"] >= date_right - DATE_WIN) & (df["right"] <= date_right + DATE_WIN)
    if not mask.any():
        return None
    txt = _norm(" ".join(df.loc[mask, "text"].astype(str).tolist()))
    # normalize separators and spaces
    txt = txt.replace(",", ".").replace("•", ".")
    m = DATE_ANY_SEP_RE.search(txt)
    if not m:
        return None
    dd, mm, yy = re.findall(r"\d+", m.group(0))[:3]
    cand = f"{dd}.{mm}.{yy}"
    return parse_date(cand) or None


def _extract_date_anywhere(text: str) -> Optional[str]:
    if not text:
        return None
    norm = (
        text.replace("•", ".")
            .replace(",", ".")
            .replace(" ", "")
    )
    m = DATE_ANYWHERE_RE.search(norm)
    if m:
        return m.group(1)
    return None


def _words_df_from_page(page: dict) -> pd.DataFrame:
    rows: List[dict] = []
    for line in page.get("lines", []) or []:
        for w in line.get("words", []) or []:
            rows.append(w)
    if not rows:
        return pd.DataFrame(columns=["text", "left", "top", "width", "height", "right", "bottom"])
    df = pd.DataFrame(rows).copy()
    for c in ("left", "top", "width", "height"):
        if c in df.columns:
            df[c] = df[c].astype(int)
    df["right"] = df["left"] + df["width"]
    df["bottom"] = df["top"] + df["height"]
    df["text"] = df["text"].astype(str)
    return df

def _find_header_positions_in_page(page: dict) -> Optional[Dict[str, int]]:
    best = {"desc_left": None, "date_left": None, "date_right": None, "amount_left": None, "amount_right": None}
    for line in page.get("lines", []) or []:
        words = line.get("words", []) or []
        if not words:
            continue
        texts = [w.get("text", "").strip().lower() for w in words]

        # Description
        for i, t in enumerate(texts):
            if t == "description" and best["desc_left"] is None:
                w = words[i]; best["desc_left"] = int(w["left"])

        # Booking Date (two tokens)
        for i, t in enumerate(texts):
            if t == "booking" and best["date_left"] is None:
                w = words[i]; best["date_left"] = int(w["left"])
            if t == "date":
                w = words[i]
                best["date_right"] = int(w["left"]) + int(w["width"])
                if best["date_left"] is None:
                    best["date_left"] = int(w["left"])

        # Amount
        for i, t in enumerate(texts):
            if t == "amount":
                w = words[i]
                best["amount_left"]  = int(w["left"])
                best["amount_right"] = int(w["left"]) + int(w["width"])
    if best["amount_right"] is None:
        return None
    return best

def _detect_columns_from_headers(raw_pages: List[dict]) -> Dict[str, int]:
    hits: List[Dict[str, int]] = []
    for p in raw_pages[:5]:
        pos = _find_header_positions_in_page(p)
        if pos:
            hits.append(pos)
    # collect a rough page width
    max_right = 0
    for p in raw_pages:
        for ln in p.get("lines", []) or []:
            for w in ln.get("words", []) or []:
                r = int(w.get("left", 0)) + int(w.get("width", 0))
                if r > max_right: max_right = r
    page_right = max_right or 2200

    if not hits:
        cols = {"desc_left": 100, "date_left": -1, "date_right": -1, "amount_left": -1, "amount_right": -1,
                "desc_max_right": int(page_right * 0.55)}
        if DEBUG: print(f"[N26] GLOBAL COLUMNS (no-header): {cols}")
        return cols

    def med(key: str) -> Optional[int]:
        vals = [h[key] for h in hits if h.get(key) is not None]
        return int(pd.Series(vals).median()) if vals else None

    desc_left = med("desc_left")
    date_left = med("date_left")
    date_right = med("date_right")
    amount_left = med("amount_left")
    amount_right = med("amount_right")

    # ✅ Use date_right to define where description ends (safer),
    #    else fall back to amount_right - 300.
    if date_right is not None:
        desc_max_right = max(0, int(date_right) - 20)
    elif amount_right is not None:
        desc_max_right = max(0, int(amount_right) - 300)
    else:
        desc_max_right = int(page_right * 0.55)

    cols = {
        "desc_left": desc_left if desc_left is not None else 100,
        "date_left": -1 if date_left is None else int(date_left),
        "date_right": -1 if date_right is None else int(date_right),
        "amount_left": -1 if amount_left is None else int(amount_left),
        "amount_right": -1 if amount_right is None else int(amount_right),
        "desc_max_right": int(desc_max_right),
    }
    if DEBUG: print(f"[N26] GLOBAL COLUMNS: {cols}")
    return cols


# ---------- Tiny helpers ----------


# put this near other regexes
DATE_ANY_SEP_RE = re.compile(r"\b\d{1,2}\D{1,3}\d{1,2}\D{1,3}\d{2,4}\b")

def _amount_from_tokens(tokens: List[str]) -> Optional[float]:
    if not tokens:
        return None
    s = _norm(" ".join(t for t in tokens if t and str(t).strip()))
    # 🚫 remove date substrings so "20.08.2024" can't become "20.08"
    s = DATE_ANY_SEP_RE.sub(" ", s)
    m = AMOUNT_CHUNK_REGEX.search(s)
    if not m:
        return None
    v = parse_currency(m.group(0), strip_currency=False)
    return float(v) if pd.notna(v) else None



def _clean_desc_line(s: str) -> str:
    if not s: return ""
    s = VALUE_DATE_RE.sub("", s)
    s = FOOTER_RE.sub("", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip(" •,-–—\t")

# ---------- Core ----------
def parse_n26_transactions_single_pass(pages: List[dict]) -> List[dict]:
    """
    - Columns from header words (independent of vertical alignment).
    - Primary anchors: (amount near amount column) AND (date anywhere on the line).
    - Fallback anchors (if none on a page): (date anywhere) AND (amount anywhere to the right of desc_max_right + 40).
    - Description from lines between anchors, using only words left of desc_max_right.
    """
    results: List[dict] = []
    seq = 0
    currency = "EUR"

    cols = _detect_columns_from_headers(pages)
    desc_max_right = int(cols["desc_max_right"])
    date_right  = int(cols.get("date_right", -1))
    amount_right= int(cols.get("amount_right", -1))

    for p_idx, page in enumerate(pages):
        lines = page.get("lines", []) or []
        if not lines:
            if DEBUG: print(f"[N26] page {p_idx+1}: no lines")
            continue

        line_dfs: List[pd.DataFrame] = []
        for ln in lines:
            w = ln.get("words", []) or []
            df = pd.DataFrame(w)
            if df.empty:
                df = pd.DataFrame(columns=["text", "left", "top", "width", "height", "right", "bottom"])
            else:
                for c in ("left", "top", "width", "height"):
                    if c in df.columns:
                        df[c] = df[c].astype(int)
                df["right"] = df["left"] + df["width"]
                df["bottom"] = df["top"] + df["height"]
                df["text"] = df["text"].astype(str)
            line_dfs.append(df)

        anchors: List[Dict[str, Any]] = []

        # ---- Pass A: collect date lines & amount lines separately ----
        date_hits: List[Dict[str, Any]] = []
        amt_hits: List[Dict[str, Any]]  = []

        for idx, (ln, df) in enumerate(zip(lines, line_dfs)):
            lt = _norm((ln.get("line_text") or "").strip())
            if not lt:
                continue

            # For DATE discovery: do NOT skip "Value Date ..." lines.
            skip_for_amount = SUMMARY_ROW_RE.match(lt) or VALUE_DATE_RE.search(lt)

            if DEBUG:
                lt = _norm((ln.get("line_text") or "").strip())
                if any(ch.isdigit() for ch in lt):
                    if '.' in lt or '/' in lt or '-' in lt:
                        print(f"[DEBUG DATE CANDIDATE] {lt!r}")

            # --- date anywhere on the line ---
            found_date = _extract_date_from_df_col(df, date_right)
            if found_date:
                date_hits.append({"idx": idx, "date": found_date})

            # --- amount: only if this line isn't a Value Date/Summary line ---
            found_amt = None
            if not skip_for_amount:
                if amount_right >= 0 and not df.empty:
                    mask_a = (df["right"] >= amount_right - AMOUNT_WIN) & (df["right"] <= amount_right + AMOUNT_WIN)
                    tokens = df.loc[mask_a, "text"].astype(str).tolist()
                    found_amt = _amount_from_tokens(tokens)
                    if found_amt is None:
                        mask2 = (df["right"] >= amount_right - (AMOUNT_WIN + 20)) & (
                                    df["right"] <= amount_right + (AMOUNT_WIN + 20))
                        tokens2 = df.loc[mask2, "text"].astype(str).tolist()
                        found_amt = _amount_from_tokens(tokens2)

                if found_amt is None and not df.empty:
                    right_floor = max(desc_max_right + 20, 600)
                    cand = df[df["right"] >= right_floor]
                    found_amt = _amount_from_tokens(cand["text"].astype(str).tolist()) if not cand.empty else None

                if found_amt is None:
                    found_amt = _amount_from_tokens([lt])

            if found_amt is not None:
                amt_hits.append({"idx": idx, "amount": float(found_amt)})

        # ---- Pass B: pair amount lines with the nearest previous date line (within small gap) ----
        # Allow date and amount to be on separate consecutive lines (gap<=2).
        max_gap = 2
        di = 0
        for ah in amt_hits:
            i = ah["idx"]
            # move date pointer up to the last date line not after i
            while di < len(date_hits) and date_hits[di]["idx"] <= i:
                di += 1
            # candidates are date_hits[0..di-1]; pick the last one within gap
            chosen = None
            for dj in range(di - 1, -1, -1):
                j = date_hits[dj]["idx"]
                if 0 <= (i - j) <= max_gap:
                    chosen = date_hits[dj]
                    break
                if j < i - max_gap:
                    break
            if chosen:
                anchors.append({"idx": i, "date": chosen["date"], "amount": ah["amount"]})

        if DEBUG:
            print(f"[N26] page {p_idx+1}: date_hits={len(date_hits)} amt_hits={len(amt_hits)} paired_anchors={len(anchors)}")
            for a in anchors[:3]:
                print(f"       · anchor idx={a['idx']} date={a['date']} amount={a['amount']}")


        if not anchors:
            continue

        # ---- Descriptions between anchors ----
        prev_idx = -1

        def left_of_cut_text(_ln: dict, _df: pd.DataFrame) -> str:
            if _df.empty: return ""
            toks = _df[_df["right"] <= desc_max_right]["text"].astype(str).tolist()
            if toks:
                return " ".join(toks).strip()
            return (_ln.get("line_text") or "").strip()

        for a in anchors:
            start = prev_idx + 1
            end   = a["idx"]
            desc_lines: List[str] = []

            for k in range(start, end + 1):
                txt = left_of_cut_text(lines[k], line_dfs[k])
                txt = _clean_desc_line(txt)
                if not txt:
                    continue
                if SUMMARY_ROW_RE.match(txt) or txt.lower() == "description":
                    continue
                if DATE_DOT_REGEX.fullmatch(txt) or AMOUNT_CHUNK_REGEX.fullmatch(txt):
                    continue
                desc_lines.append(txt)

            # Merchant line = first meaningful line; fallback to joined unique lines
            description = ""
            for dl in desc_lines:
                if dl.lower().startswith(("value date", "previous", "incoming", "outgoing")):
                    continue
                description = dl
                break
            if not description and desc_lines:
                seen, uniq = set(), []
                for dl in desc_lines:
                    if dl in seen: continue
                    seen.add(dl); uniq.append(dl)
                description = " ".join(uniq)
            description = FOOTER_RE.sub("", description).strip()

            results.append({
                "seq": seq,
                "transactions_date": a["date"],
                "transaction_type": "debit" if a["amount"] < 0 else "credit",
                "description": description,
                "amount": {"value": abs(a["amount"]), "currency": currency},
                "balance_after_statement": None,
                "balance_after_calculated": None,
            })
            seq += 1
            prev_idx = end

    for i, r in enumerate(results):
        r["seq"] = i
    if DEBUG: print(f"[N26] total transactions parsed: {len(results)}")
    return results

# ---------- Currency bucketing & entrypoint ----------
def _group_by_currency(transactions: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in transactions:
        cur = t.get("amount", {}).get("currency")
        if cur: buckets[cur].append(t)
    for cur in buckets:
        buckets[cur].sort(key=lambda t: t.get("seq", 0))
    return buckets

def _build_currency_sections_from_rows(buckets: Dict[str, List[dict]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cur, txs in sorted(buckets.items()):
        money_in  = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit"), 2)
        out[cur] = {
            "opening_balance": None,
            "money_in_total":  {"value": money_in,  "currency": cur},
            "money_out_total": {"value": money_out, "currency": cur},
            "closing_balance_statement": None,
            "closing_balance_calculated": None,
            "transactions": txs,
        }
    return out

def extract_iban(pages: List[dict]) -> Optional[str]:
    iban_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b')
    for page in pages:
        for line in page.get("lines", []):
            txt = (line.get("line_text") or "")
            if "IBAN" in txt.upper():
                after = txt.upper().split("IBAN", 1)[-1]
                cleaned = re.sub(r"\s+", "", after)
                m = iban_pattern.search(cleaned)
                if m: return m.group(1)
    return None

def extract_bic(pages: List[dict]) -> Optional[str]:
    bic_token = re.compile(r'\b([A-Z0-9]{8}([A-Z0-9]{3})?)\b')
    for page in pages:
        for line in page.get("lines", []):
            txt = (line.get("line_text") or "")
            if "BIC" in txt.upper():
                after = txt.upper().split("BIC", 1)[-1]
                after = re.sub(r"[:\s]+", " ", after).strip()
                m = bic_token.search(after)
                if m: return m.group(1)
    return None

def parse_statement(raw_ocr, client: str = "Unknown", account_type: str = "Unknown"):
    pages = raw_ocr.get("pages", []) or []

    transactions = parse_n26_transactions_single_pass(pages)
    buckets = _group_by_currency(transactions)
    currencies = _build_currency_sections_from_rows(buckets)

    iban = extract_iban(pages)
    bic  = extract_bic(pages)

    if transactions:
        dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(dates) if dates else None
        end_date   = max(dates) if dates else None
    else:
        start_date = end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,
        "institution": "N26 Bank",
        "account_type": account_type,
        "iban": iban,
        "bic": bic,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
