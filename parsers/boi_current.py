# boi.py
# Bank of Ireland parser → AIB-style output (balances + transactions)
#
# - headers: "Date | Transaction details | Payments - out | Payments - in | Balance"
# - robust header detection (hyphen/en-dash/emdash; merged/split tokens)
# - opening balance: first "BALANCE FORWARD" row's Balance value
# - closing balance: last number near Balance column
# - descriptions: tokens from the Details column window only
# - output:
#   currencies: {
#     <CUR>: {
#       balances: {
#         opening_balance: { summary_table: None, transactions_table: <float|None> },
#         money_in_total:  { summary_table: None, transactions_table: <float> },
#         money_out_total: { summary_table: None, transactions_table: <float> },
#         closing_balance: { summary_table: None, transactions_table: <float|None>, calculated: <float|None> }
#       },
#       transactions: [ { seq, transaction_date, transaction_type, description, amount } ... ]
#     }
#   }

from __future__ import annotations

import hashlib
import re
from typing import Optional, Dict, List, Any

import pandas as pd

from utils import parse_currency, parse_date

# ---------- Regex helpers ----------
RE_BAL_FWD = re.compile(r"\bBALANCE\s*FORWARD\b", re.IGNORECASE)
RE_DATE_LONG = re.compile(r"\b(\d{1,2})\s+[A-Za-z]{3,9}\s+\d{4}\b")
MONEY_RE = re.compile(r"^\d{1,3}(?:,\d{3})*\.\d{2}$|^\d+\.\d{2}$")

DASH_CHARS = "-–—"  # hyphen / en dash / em dash

def _find_date_in_text(s: str) -> Optional[str]:
    if not s:
        return None
    m = RE_DATE_LONG.search(s)
    if m:
        return parse_date(m.group(0))
    return None

# ---------- Column detection ----------
def detect_column_positions(lines: list[dict], debug: bool = False) -> tuple[dict, int, int] | None:
    """
    Detect BOI headers and return:
      positions = {
        'date_left': int,
        'details_left': int,
        'out_right': int,
        'in_right': int,
        'bal_right': int,
      }, header_start_idx, len(lines)
    """
    def right_edge(w: dict) -> int:
        return int(w.get("left", 0)) + int(w.get("width", 0))

    def dbg(*a):
        if debug:
            print(*a)

    for idx, line in enumerate(lines or []):
        words = line.get("words", []) or []
        if not words:
            continue

        # Prepare normalized token list
        toks = [(i, (w.get("text") or "").strip(), int(w.get("left", 0)), right_edge(w)) for i, w in enumerate(words)]
        toks_lower = [(i, t.lower(), l, r) for (i, t, l, r) in toks if t]

        pos = {
            "date_left": None,
            "details_left": None,
            "out_right": None,
            "in_right": None,
            "bal_right": None,
        }

        # Find "Date"
        for i, t, l, r in toks_lower:
            if t == "date":
                pos["date_left"] = int(l)
                break

        # Find "Transaction details" (may come as "Transaction" then "details")
        for i, t, l, r in toks_lower:
            if t.startswith("transaction"):
                # if next token is "details" use this left as the column left
                if i + 1 < len(words):
                    nxt = (words[i + 1].get("text") or "").strip().lower()
                    if "detail" in nxt:
                        pos["details_left"] = int(l)
                        break
                # sometimes it is a single merged token
                if "detail" in t:
                    pos["details_left"] = int(l)
                    break

        # Find Payments - out / in (robust to dash variants and merged tokens)
        def find_payments(label: str) -> Optional[int]:
            """
            label in {"out","in"}; returns RIGHT edge of the label token (or the last token
            in the header span) for the respective Payments column.
            """
            for i, t, l, r in toks_lower:
                if not t.startswith("payments"):
                    continue
                # Look ahead up to 4 tokens for dash + label (or merged)
                span_right = r
                found = False
                j = i + 1
                steps = 0
                while j < len(words) and steps < 5:
                    tj = (words[j].get("text") or "").strip().lower()
                    span_right = right_edge(words[j])
                    if (tj in label) or (tj == label):
                        found = True
                        break
                    # tolerate dash tokens and merged forms
                    if (tj and any(ch in tj for ch in DASH_CHARS)) or (("payments" + label) in t.replace(" ", "")) or (label in tj):
                        # if the very next meaningful token contains label, accept
                        if label in tj:
                            found = True
                            break
                    j += 1
                    steps += 1
                if found:
                    return int(span_right)
            return None

        pos["out_right"] = find_payments("out")
        pos["in_right"]  = find_payments("in")

        # Find "Balance"
        for i, t, l, r in toks_lower:
            if "balance" in t:
                pos["bal_right"] = int(r)
                break

        if all(pos[k] is not None for k in ("date_left", "details_left", "out_right", "in_right", "bal_right")):
            dbg(f"🔎 BOI header detected at line {idx}: {pos}")
            return pos, idx, len(lines)

    return None


def categorise_amount_by_right_edge(x_right: int, header_positions: dict, margin: int = 140) -> str:
    """
    Classify token into 'out' | 'in' | 'bal' by proximity to column RIGHT edges.
    """
    x_right = int(x_right)
    best = ("unknown", margin + 1)
    for k, label in (("out_right", "out"), ("in_right", "in"), ("bal_right", "bal")):
        pos = header_positions.get(k)
        if pos is None:
            continue
        d = abs(x_right - int(pos))
        if d < best[1]:
            best = (label, d)
    return best[0] if best[1] <= margin else "unknown"

# ---------- Geometry helpers ----------
def _right_edge(w: dict) -> int:
    return int(w.get("left", 0)) + int(w.get("width", 0))

def _details_window_words(words: list[dict], details_left: int, amount_cols_right: List[int]) -> List[str]:
    """
    Return token texts within the Details column by geometry:
    left >= details_left - pad  AND  right <= desc_right_limit,
    where desc_right_limit is slightly left of the first amount column.
    """
    PAD_LEFT = 6
    GAP_RIGHT = 30  # buffer before the first amount column
    desc_right_limit = min(amount_cols_right) - GAP_RIGHT

    out = []
    for w in words or []:
        txt = (w.get("text") or "").strip()
        if not txt:
            continue
        l = int(w.get("left", 0))
        r = _right_edge(w)
        if (l >= details_left - PAD_LEFT) and (r <= desc_right_limit):
            out.append(txt)
    return out

# ---------- Opening & Closing ----------
def extract_opening_balance_and_start_date(pages: list[dict], debug: bool = False) -> tuple[float | None, str | None]:
    """
    Opening = balance value of the FIRST 'BALANCE FORWARD' line (nearest the Balance column).
    Start date = date on that same line (if present).
    """
    def dbg(*a):
        if debug:
            print(*a)

    for page_idx, page in enumerate(pages or []):
        det = detect_column_positions(page.get("lines", []) or [], debug=debug)
        bal_r = None
        if det:
            bal_r = int(det[0]["bal_right"])

        for line_idx, line in enumerate(page.get("lines", []) or []):
            lt = (line.get("line_text") or "")
            if not RE_BAL_FWD.search(lt):
                continue

            df = pd.DataFrame(line.get("words", []) or [])
            if df.empty:
                continue
            df["right"] = df["left"] + df["width"]
            df["amount_val"] = df["text"].apply(lambda v: parse_currency(v, strip_currency=False))
            df_num = df[df["amount_val"].notna()].copy()
            if df_num.empty:
                continue

            if bal_r is not None:
                df_num["dist"] = (df_num["right"] - bal_r).abs()
                chosen = df_num.sort_values(["dist", "right"]).iloc[0]
                opening = float(chosen["amount_val"])
                dbg(f"📘 Opening from page {page_idx} line {line_idx}: {opening} (by Balance col)")
            else:
                # fallback: take rightmost numeric on the line
                chosen = df_num.sort_values("right").iloc[-1]
                opening = float(chosen["amount_val"])
                dbg(f"📘 Opening from page {page_idx} line {line_idx}: {opening} (fallback rightmost)")

            start = _find_date_in_text(lt)
            return opening, start

    return None, None

def extract_closing_balance(pages: list[dict], debug: bool = False) -> Optional[float]:
    """
    Closing balance = the last token in the Balance column that looks like a real money amount.
    We:
      - restrict to lines after a detected header on each page,
      - classify tokens via right-edge proximity (so we're inside the Balance column),
      - and require a 2-decimal money shape (rejects stray integers like '4').
    """
    def dbg(*a):
        if debug:
            print(*a)

    best = None

    for pidx, page in enumerate(pages or []):
        det = detect_column_positions(page.get("lines", []) or [], debug=debug)
        if not det:
            continue
        pos, start_idx, end_idx = det
        bal_r = int(pos["bal_right"])

        for line in (page.get("lines", []) or [])[start_idx + 1 : end_idx]:
            words = line.get("words", []) or []
            if not words:
                continue

            # Build a small frame: token text, right edge, parsed value
            df = pd.DataFrame(words)
            if df.empty:
                continue
            df["right"] = df["left"] + df["width"]
            df["text_norm"] = df["text"].astype(str).str.strip()
            # classify by column right-edges
            df["category"] = df["right"].apply(lambda xr: categorise_amount_by_right_edge(int(xr), pos, margin=140))

            # candidates: Balance column AND money-like with 2 decimals
            cand = df[(df["category"] == "bal") & (df["text_norm"].str.match(MONEY_RE))]
            if cand.empty:
                continue

            # pick the rightmost candidate on the line (visual end of the column)
            tok = cand.iloc[cand["right"].argmax()]
            v = parse_currency(tok["text_norm"], strip_currency=False)
            if v is None:
                continue

            best = float(v)  # last wins across the statement

            if debug:
                dbg(f"🧾 Closing cand p{pidx}: '{tok['text_norm']}' → {best:.2f}  (xr={int(tok['right'])}, bal_r={bal_r})")

    return best

# ---------- Transactions ----------
def parse_transactions(pages: list[dict], debug: bool = False) -> list[dict]:
    all_transactions: List[dict] = []
    seq = 0

    for pidx, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        det = detect_column_positions(lines, debug=debug)
        if not det:
            continue
        pos, start_idx, end_idx = det
        last_seen_date: Optional[str] = None

        for lidx, line in enumerate(lines[start_idx + 1 : end_idx]):
            words = line.get("words", []) or []
            if not words:
                continue

            ltxt = line.get("line_text", "") or ""
            d = _find_date_in_text(ltxt)
            if d:
                last_seen_date = d
            if last_seen_date is None:
                continue

            # Skip the opening row itself
            if RE_BAL_FWD.search(ltxt):
                continue

            df = pd.DataFrame(words)
            if df.empty:
                continue
            df["right"] = df["left"] + df["width"]
            df["amount_val"] = df["text"].apply(lambda v: parse_currency(v, strip_currency=False))
            df["category"] = df["right"].apply(lambda xr: categorise_amount_by_right_edge(int(xr), pos))

            out_series = df[(df["amount_val"].notna()) & (df["category"] == "out")]["amount_val"]
            in_series  = df[(df["amount_val"].notna()) & (df["category"] == "in")]["amount_val"]

            debit  = float(out_series.iloc[-1]) if not out_series.empty else 0.0
            credit = float(in_series .iloc[-1]) if not in_series .empty else 0.0
            if debit == 0.0 and credit == 0.0:
                continue

            # Description from Details window
            details_tokens = _details_window_words(
                words,
                details_left=int(pos["details_left"]),
                amount_cols_right=[int(pos["out_right"]), int(pos["in_right"]), int(pos["bal_right"])],
            )
            clean_desc = " ".join(details_tokens).strip()

            all_transactions.append({
                "seq": seq,
                "transaction_date": last_seen_date,
                "transaction_type": "credit" if credit > 0 else "debit",
                "description": clean_desc,
                "amount": credit if credit > 0 else debit,
                "signed_amount": (credit if credit > 0 else -debit),
            })

            seq += 1

    return all_transactions

# ---------- IBAN & currency ----------
def extract_iban(pages: list[dict]) -> str | None:
    """
    Very permissive IBAN pull (used only to pick currency).
    """
    iban_pat = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b")
    for page in pages or []:
        for line in page.get("lines", []) or []:
            txt = (line.get("line_text") or "").upper()
            if "IBAN" in txt:
                cleaned = re.sub(r"\s+", "", txt.split("IBAN", 1)[-1])
                m = iban_pat.search(cleaned)
                if m:
                    return m.group(1)
    return None

def _bucket_currency(transactions: List[dict], iban: str | None) -> Dict[str, List[dict]]:
    cur = "GBP" if (iban and iban.upper().startswith("GB")) else "EUR"
    return {cur: sorted(transactions, key=lambda t: t.get("seq", 0))}

# ---------- Assemble balances structure ----------
def _build_balances(buckets: Dict[str, List[dict]], opening: float | None, closing_stmt: float | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cur, txs in buckets.items():
        money_in  = round(sum(float(t["amount"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        money_out = round(sum(float(t["amount"]) for t in txs if t.get("transaction_type") == "debit"), 2)
        closing_calc = round(opening + money_in - money_out, 2) if opening is not None else None

        out[cur] = {
            "balances": {
                "opening_balance": {
                    "summary_table": None,
                    "transactions_table": opening,
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
                    "transactions_table": closing_stmt,
                    "calculated": closing_calc,
                },
            },
            "transactions": txs,
        }
    return out

# ---------- Public entrypoint ----------
def parse_statement(raw_ocr: dict, client: str = "Unknown", account_type: str = "Unknown", debug: bool = False) -> dict:
    """
    Returns a single 'statement node' (no top-level 'client'), matching the AIB shape.
    This lets main.py call parse_statement repeatedly (across banks) and bundle all
    statements into one JSON.
    """
    pages = raw_ocr.get("pages", []) or []

    # IBAN (used to bucket currency)
    iban = extract_iban(pages)

    # Transactions + opening/closing
    transactions = parse_transactions(pages, debug=debug)
    opening_val, start_date_open = extract_opening_balance_and_start_date(pages, debug=debug)
    closing_val = extract_closing_balance(pages, debug=debug)

    # Currency bucket + balances structure (same layout as AIB)
    buckets = _bucket_currency(transactions, iban)
    currencies = _build_balances(buckets, opening=opening_val, closing_stmt=closing_val)

    # Statement date span
    if transactions:
        all_dates = [t.get("transaction_date") for t in transactions if t.get("transaction_date")]
        start_date = start_date_open or (min(all_dates) if all_dates else None)
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = start_date_open
        end_date = None

    # Optional lightweight statement_id (mirrors AIB approach)
    sid_basis = f"{raw_ocr.get('file_name') or ''}|{start_date or ''}|{end_date or ''}"
    statement_id = hashlib.sha1(sid_basis.encode("utf-8")).hexdigest()[:12] if sid_basis.strip("|") else None

    return {
        "statement_id": statement_id,
        "file_name": raw_ocr.get("file_name"),
        "institution": "Bank of Ireland",
        "account_type": account_type,
        "iban": iban,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
