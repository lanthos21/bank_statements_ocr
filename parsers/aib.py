# aib_current.py
# AIB parser:
# - Header: "Date | Details | Debit € | Credit € | Balance €"
# - Amounts are RIGHT-JUSTIFIED under the three amount columns
# - Robust opening balance ("BALANCE FORWARD …") and closing balance (last Balance €)
# - Transactions shaped like Revolut/BOI JSON (no per-row balances)

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

from utils import parse_currency, parse_date, date_variants

# ----------------------------
# Regexes
# ----------------------------

RE_BAL_FWD = re.compile(r"\bBALANCE\s*FORWARD\b", re.IGNORECASE)
RE_DATE_LONG = re.compile(r"\b(\d{1,2})\s+[A-Za-z]{3,9}\s+\d{4}\b")
RE_DATE_NUM  = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")


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
# Header / column detection (AIB)
# ----------------------------

def detect_column_positions(lines: list[dict]) -> tuple[dict, int, int] | None:
    """
    Locate AIB headers "Date | Details | Debit € | Credit € | Balance €".
    Returns (header_positions, header_start_idx, len(lines)) or None.
    """
    def right_edge(w: dict) -> int:
        return int(w.get("left", 0)) + int(w.get("width", 0))

    for idx, line in enumerate(lines or []):
        words = line.get("words", []) or []
        if not words:
            continue

        pos = {"date_left": None, "details_left": None, "debit_right": None, "credit_right": None, "balance_right": None}

        n = len(words)
        for i, w in enumerate(words):
            t = (w.get("text") or "").strip().lower()
            if not t:
                continue

            if t == "date":
                pos["date_left"] = int(w.get("left", 0))
            elif t == "details":
                pos["details_left"] = int(w.get("left", 0))
            elif t == "debit":
                pos["debit_right"] = right_edge(words[i + 1]) if i + 1 < n and (words[i + 1].get("text") or "").strip() == "€" else right_edge(w)
            elif t == "credit":
                pos["credit_right"] = right_edge(words[i + 1]) if i + 1 < n and (words[i + 1].get("text") or "").strip() == "€" else right_edge(w)
            elif t == "balance":
                pos["balance_right"] = right_edge(words[i + 1]) if i + 1 < n and (words[i + 1].get("text") or "").strip() == "€" else right_edge(w)

        if all(pos.values()):
            return pos, idx, len(lines)

    return None


def categorise_amount_by_right_edge(x_right: int, header_positions: dict, margin: int = 120) -> str:
    """Classify token into debit/credit/balance by proximity to the RIGHT edges."""
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
# Helpers for amounts on a line
# ----------------------------

def _right_edge(w: dict) -> int:
    return int(w.get("left", 0)) + int(w.get("width", 0))


def _rightmost_numeric_with_x(words: list[dict]) -> tuple[Optional[int], Optional[float]]:
    """Return (x_right, value) for the numeric whose RIGHT edge is farthest right."""
    best = None
    tokens = words or []
    n = len(tokens)

    for i, w in enumerate(tokens):
        t = (w.get("text") or "").strip()
        if not t:
            continue
        v = parse_currency(t, strip_currency=False)
        xr = _right_edge(w)
        if v is None and i + 1 < n and (tokens[i + 1].get("text") or "").strip() == "€":
            v = parse_currency(t + "€", strip_currency=False)
            xr = _right_edge(tokens[i + 1])
        if v is None:
            continue
        if best is None or xr > best[0]:
            best = (xr, float(v))
    return (None, None) if best is None else best

# ----------------------------
# Opening & Closing balances
# ----------------------------

def extract_opening_balance_and_start_date(pages: list[dict], debug: bool = False) -> tuple[float | None, str | None]:
    """
    Opening balance should come from the very first 'BALANCE FORWARD' line in the statement,
    not from later repeated headers on subsequent pages.
    """
    def dbg(*a):
        if debug:
            print(*a)

    def _num_candidates(words: list[dict]) -> list[dict]:
        out = []
        tokens = words or []
        n = len(tokens)
        for i, w in enumerate(tokens):
            txt = (w.get("text") or "").strip()
            if not txt:
                continue
            v = parse_currency(txt, strip_currency=False)
            xl = int(w.get("left", 0))
            xr = xl + int(w.get("width", 0))

            # join with a trailing '€'
            if v is None and i + 1 < n:
                nxt = (tokens[i + 1].get("text") or "").strip()
                if nxt == "€":
                    vv = parse_currency(txt + nxt, strip_currency=False)
                    if vv is not None:
                        xr = int(tokens[i + 1].get("left", 0)) + int(tokens[i + 1].get("width", 0))
                        v = vv

            if v is not None:
                out.append({
                    "text": txt,
                    "value": float(v),
                    "left": xl,
                    "right": xr,
                })
        return out

    opening_value: Optional[float] = None
    earliest_date: Optional[str] = None

    for pidx, page in enumerate(pages or []):
        for i, line in enumerate(page.get("lines", []) or []):
            lt = line.get("line_text") or ""
            if not RE_BAL_FWD.search(lt):
                continue

            # already found an opening balance → don’t overwrite it
            if opening_value is not None:
                return opening_value, earliest_date

            dbg(f"\n🔎 Page {pidx}, line {i} contains 'BALANCE FORWARD': {lt!r}")

            words = line.get("words", []) or []
            cands = _num_candidates(words)

            val = None
            if cands:
                chosen = max(cands, key=lambda d: d["right"])  # rightmost
                val = chosen["value"]
                dbg(f"    ✅ Chosen val={val:.2f} from token={chosen['text']!r}")
            elif i + 1 < len(page.get("lines", [])):
                # peek next line if no number on same line
                next_words = page["lines"][i + 1].get("words", []) or []
                next_cands = _num_candidates(next_words)
                if next_cands:
                    chosen = max(next_cands, key=lambda d: d["right"])
                    val = chosen["value"]
                    dbg(f"    ✅ Chosen val={val:.2f} from NEXT line token={chosen['text']!r}")

            d = _find_date_in_text(lt)
            if d:
                dbg(f"    📅 Date detected: {d}")

            if val is not None:
                opening_value = val
                earliest_date = d
                dbg(f"    ⬆️ Setting opening balance {opening_value:.2f} at date {earliest_date}")
                return opening_value, earliest_date  # stop at first match

    dbg("⚠️ No opening balance found")
    return None, None


def extract_closing_balance(pages: list[dict]) -> Optional[float]:
    """Closing balance = last number near Balance € column."""
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

# ----------------------------
# Transactions (AIB)
# ----------------------------

def parse_transactions(pages: list[dict], iban: str | None = None) -> list[dict]:
    all_transactions = []
    current_currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    seq = 0
    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_positions(lines)
        if not header:
            continue
        header_positions, start_idx, end_idx = header
        last_seen_date = None
        for line in lines[start_idx + 1: end_idx]:
            ltxt = line.get("line_text", "") or ""
            if RE_BAL_FWD.search(ltxt):
                d = _find_date_in_text(ltxt)
                if d:
                    last_seen_date = d
                continue
            words = line.get("words", []) or []
            if not words:
                continue
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
            debit  = float(debit_vals.sort_values("right")["amount_val"].iloc[-1]) if not debit_vals.empty else 0.0
            credit = float(credit_vals.sort_values("right")["amount_val"].iloc[-1]) if not credit_vals.empty else 0.0
            if debit == 0.0 and credit == 0.0:
                continue
            # Start with raw text
            clean_desc = ltxt.strip()

            # Remove the known transaction date in any common printed form
            if last_seen_date:
                for variant in date_variants(last_seen_date):
                    if variant in clean_desc:
                        clean_desc = clean_desc.replace(variant, "")

            # Also strip trailing currency amounts/symbols
            for w in words:
                t = (w.get("text") or "").strip()
                if parse_currency(t, strip_currency=False) is not None or t == "€":
                    clean_desc = clean_desc.replace(t, "")

            clean_desc = clean_desc.strip(" |-")

            all_transactions.append({
                "seq": seq,
                "transactions_date": last_seen_date,
                "transaction_type": "credit" if credit > 0 else "debit",
                "description": clean_desc,
                "amount": {"value": credit if credit > 0 else debit, "currency": current_currency},
            })
            seq += 1
    return all_transactions

# ----------------------------
# Buckets and per-currency sections
# ----------------------------

def _group_by_currency(transactions: List[dict]) -> Dict[str, List[dict]]:
    buckets = defaultdict(list)
    for t in transactions:
        cur = t.get("amount", {}).get("currency")
        if cur:
            buckets[cur].append(t)
    for cur in buckets:
        buckets[cur].sort(key=lambda t: t.get("seq", 0))
    return buckets


def _build_currency_sections_from_rows(buckets: Dict[str, List[dict]]) -> Dict[str, Any]:
    out = {}
    for cur, txs in buckets.items():
        money_in  = round(sum(float(t["amount"]["value"]) for t in txs if t["transaction_type"] == "credit"), 2)
        money_out = round(sum(float(t["amount"]["value"]) for t in txs if t["transaction_type"] == "debit"), 2)
        out[cur] = {
            "opening_balance": None,
            "money_in_total":  {"value": money_in, "currency": cur},
            "money_out_total": {"value": money_out, "currency": cur},
            "closing_balance_statement": None,
            "closing_balance_calculated": None,
            "transactions": txs,
        }
    return out

# ----------------------------
# Public entrypoint (AIB)
# ----------------------------

def extract_iban(pages: list[dict]) -> str | None:
    iban_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b')
    for page in pages:
        for line in page.get("lines", []):
            txt = line.get("line_text", "") or ""
            if "IBAN" in txt.upper():
                m = iban_pattern.search(re.sub(r"\s+", "", txt.upper().split("IBAN",1)[-1]))
                if m:
                    return m.group(1)
    return None


def parse_statement(raw_ocr, client="Unknown", account_type="Unknown", debug: bool = True):
    def dbg(*a):
        if debug: print(*a)

    pages = raw_ocr.get("pages", []) or []
    iban = extract_iban(pages)
    currency_hint = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    transactions = parse_transactions(pages, iban=iban)
    dbg(f"🧾 Parsed {len(transactions)} transactions")
    buckets = _group_by_currency(transactions)
    currencies = _build_currency_sections_from_rows(buckets)

    opening_val, start_date = extract_opening_balance_and_start_date(pages)
    closing_val = extract_closing_balance(pages)
    dbg(f"📘 Explicit opening_val={opening_val} start_date={start_date}")
    dbg(f"📘 Explicit closing_val={closing_val}")

    target_cur = "EUR" if "EUR" in currencies else next(iter(currencies.keys()), currency_hint)
    if target_cur not in currencies:
        currencies[target_cur] = {"opening_balance": None,
                                  "money_in_total": {"value": 0.0, "currency": target_cur},
                                  "money_out_total": {"value": 0.0, "currency": target_cur},
                                  "closing_balance_statement": None,
                                  "closing_balance_calculated": None,
                                  "transactions": []}
    sec = currencies[target_cur]
    if opening_val is not None:
        sec["opening_balance"] = {"value": float(round(opening_val, 2)), "currency": target_cur}
    if closing_val is not None:
        sec["closing_balance_statement"] = {"value": float(round(closing_val, 2)), "currency": target_cur}
    if sec.get("opening_balance"):
        o,inm,outm = float(sec["opening_balance"]["value"]), float(sec["money_in_total"]["value"]), float(sec["money_out_total"]["value"])
        sec["closing_balance_calculated"] = {"value": round(o+inm-outm,2), "currency": target_cur}

    if transactions:
        all_dates = [t["transactions_date"] for t in transactions if t["transactions_date"]]
        start_date = start_date or (min(all_dates) if all_dates else None)
        end_date   = max(all_dates) if all_dates else None
    else:
        end_date = None

    return {"client": client,
            "file_name": raw_ocr.get("file_name"),
            "account_holder": None,
            "institution": "AIB",
            "account_type": account_type,
            "iban": iban,
            "bic": None,
            "statement_start_date": start_date,
            "statement_end_date": end_date,
            "currencies": currencies}
