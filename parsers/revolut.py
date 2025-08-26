# revolut.py
import hashlib
import re
import math
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd

from utils import parse_currency, parse_date
from lang import HEADER_SYNONYMS  # expects keys: date, description, money_out, money_in, balance

# ---------------------------------------------------------------------
# Debug knobs
# ---------------------------------------------------------------------
DEBUG_BAL = True   # print running-balance mismatches
EPS = 0.02         # tolerance when comparing floats

# ---------------------------------------------------------------------
# Header detection / geometry helpers
# ---------------------------------------------------------------------
def detect_column_positions(
    lines: List[dict],
    *,
    start_at: int = 0,
    require_balance: bool = False,
) -> tuple[dict, int, int]:
    """
    Find the transaction header row (Date / Description / Money out / Money in / Balance).
    If require_balance=True, headers that don't include a Balance column are ignored (e.g., 'Start date ...' tables).
    Returns (header_positions, header_idx, end_idx=len(lines)).
    header_positions: {"date": int|None, "desc": int|None, "out": int|None, "in": int|None, "bal": int|None}
    """
    def match_header(text: str, keywords: List[str]) -> bool:
        s = text.lower()
        return any(k in s for k in keywords)

    for idx, line in enumerate(lines[start_at:], start_at):
        line_text = " ".join((w.get("text") or "") for w in line.get("words", []))
        if (
            match_header(line_text, HEADER_SYNONYMS["date"]) and
            match_header(line_text, HEADER_SYNONYMS["description"]) and
            (match_header(line_text, HEADER_SYNONYMS["money_out"]) or
             match_header(line_text, HEADER_SYNONYMS["money_in"]))
        ):
            positions: Dict[str, Optional[int]] = {
                "date": None, "description": None, "money_out": None, "money_in": None, "balance": None
            }

            words = line.get("words", []) or []
            for i, word in enumerate(words):
                text = (word.get("text") or "").lower()
                x = int(word.get("left", word.get("x", 0)))

                if any(text == k or k in text for k in HEADER_SYNONYMS["date"]):
                    positions["date"] = x
                elif any(text == k or k in text for k in HEADER_SYNONYMS["description"]):
                    positions["description"] = x
                elif any(text == k or k in text for k in HEADER_SYNONYMS["balance"]):
                    w = int(word.get("width", 0))
                    positions["balance"] = x + w if w > 0 else x

                if i + 1 < len(words):
                    nxt = (words[i+1].get("text") or "").lower()
                    combined = f"{text} {nxt}"
                    if any(combined == k for k in HEADER_SYNONYMS["money_out"]):
                        positions["money_out"] = x
                    elif any(combined == k for k in HEADER_SYNONYMS["money_in"]):
                        positions["money_in"] = x

            # If we require a Balance column and it wasn't found, skip this header (e.g., "Start date ..." tables)
            if require_balance and positions["balance"] is None:
                continue

            return ({
                "date": positions["date"],
                "desc": positions["description"],
                "out":  positions["money_out"],
                "in":   positions["money_in"],
                "bal":  positions["balance"],
            }, idx, len(lines))

    raise ValueError("❌ Could not find transaction header row.")

def categorise_amount_by_left_edge(x1, header_positions, margin=60, width=None):
    """
    Classify an amount token into 'out' | 'in' | 'bal' by proximity to column left edges.
    For 'bal' we compare right edges if width is available.
    """
    x1 = int(x1)
    best = ("unknown", margin + 1)
    for k in ("out", "in", "bal"):
        pos = header_positions.get(k)
        if pos is None:
            continue
        d = abs((x1 + int(width)) - int(pos)) if (k == "bal" and width is not None) else abs(x1 - int(pos))
        if d < best[1]:
            best = (k, d)
    return best[0] if best[1] <= margin else "unknown"

def transaction_description(words, desc_x, first_amount_x, tol=8):
    """
    Description = tokens between (desc_x - tol) and (first_amount_x - tol).
    """
    if desc_x is None or first_amount_x is None:
        return ""
    desc_x = int(desc_x)
    first_amount_x = int(first_amount_x)

    parts: List[str] = []
    for w in sorted(words, key=lambda ww: int(ww.get("left", ww.get("x", 0)))):
        x_left = int(w.get("left", w.get("x", 0)))
        if x_left < desc_x - tol:
            continue
        if x_left >= first_amount_x - tol:
            break
        t = w.get("text", "")
        if t:
            parts.append(t)
    return " ".join(parts).strip()

def _is_total_line(text: str) -> bool:
    return text and "total" in text.lower()

def _amount_tokens_from_line(line: dict) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for w in line.get("words", []):
        t = w.get("text", "") or ""
        val = parse_currency(t, strip_currency=True)
        if val is None:
            continue
        x_left = int(w.get("left", w.get("x", 0)))
        out.append((x_left, float(val)))
    out.sort(key=lambda p: p[0])
    return out

def find_totals_row_by_columns(lines: List[dict], header_positions: dict) -> Optional[dict]:
    """
    Find a 'Total …' line. Map amounts by nearest column:
      b -> Money out   | c -> Money in   | d -> Balance
    Then compute a = d - c + b.
    """
    if not header_positions:
        return None
    out_x = header_positions.get("out")
    in_x  = header_positions.get("in")
    bal_x = header_positions.get("bal")
    if out_x is None or in_x is None or bal_x is None:
        return None

    for line in lines:
        words = line.get("words", [])
        line_text = line.get("line_text") or " ".join(w.get("text", "") for w in words)
        if not _is_total_line(line_text):
            continue

        amounts = _amount_tokens_from_line(line)
        if not amounts:
            continue

        best = {"out": (float("inf"), None), "in": (float("inf"), None), "closing": (float("inf"), None)}
        for x_left, val in amounts:
            d_out = abs(x_left - int(out_x))
            if d_out < best["out"][0]:
                best["out"] = (d_out, val)
            d_in = abs(x_left - int(in_x))
            if d_in < best["in"][0]:
                best["in"] = (d_in, val)
            d_bal = abs(x_left - int(bal_x))
            if d_bal < best["closing"][0]:
                best["closing"] = (d_bal, val)

        money_out = best["out"][1]
        money_in  = best["in"][1]
        closing   = best["closing"][1]

        opening = None
        if closing is not None and money_in is not None and money_out is not None:
            opening = round(float(closing) - float(money_in) + float(money_out), 2)

        totals = {"opening": opening, "out": money_out, "in": money_in, "closing": closing}
        if any(v is not None for v in totals.values()):
            return totals

    return None

def statement_date_range(pages: List[dict]) -> tuple[Optional[str], Optional[str]]:
    pattern = r"from (\d{1,2} \w+ \d{4}) to (\d{1,2} \w+ \d{4})"
    for page in pages:
        for line in page.get("lines", []):
            match = re.search(pattern, line.get("line_text", ""), re.IGNORECASE)
            if match:
                return match.group(1), match.group(2)
    return None, None

def _num_or_none(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v

def group_transactions_by_currency(transactions: List[dict], iban: Optional[str]) -> Dict[str, List[dict]]:
    """
    Prefer per-row currency (set in parse_transactions). If absent, fall back to a single
    bucket inferred from IBAN to keep legacy behavior.
    """
    buckets: Dict[str, List[dict]] = {}

    if any(t.get("currency") for t in transactions):
        for t in transactions:
            cur = (t.get("currency") or
                   ("GBP" if (iban and iban.upper().startswith("GB")) else "EUR"))
            buckets.setdefault(cur, []).append(t)
    else:
        # Legacy single-bucket fallback
        cur = "GBP" if (iban and iban.upper().startswith("GB")) else "EUR"
        buckets[cur] = list(transactions)

    # Keep stable ordering
    for cur in buckets:
        buckets[cur].sort(key=lambda t: t.get("seq", 0))

    return buckets


def _simple_totals_from_total_line(line: dict) -> dict | None:
    words = line.get("words", []) or []
    amts = []
    for w in words:
        v = parse_currency(w.get("text", ""), strip_currency=True)
        if v is None:
            continue
        x = int(w.get("left", w.get("x", 0)))
        amts.append((x, float(v)))
    if len(amts) < 4:
        return None
    amts.sort(key=lambda t: t[0])
    _, b, c, d = [v for _, v in amts[:4]]
    a = round(float(d) - float(c) + float(b), 2)
    return {"opening": a, "out": b, "in": c, "closing": d}

def collect_currency_summaries_from_totals(pages: list[dict], infer_page_currency) -> dict[str, dict]:
    summaries: dict[str, dict] = {}
    prev_cur: str | None = None
    for page in pages:
        cur = infer_page_currency(page, prev_cur)
        prev_cur = cur

        lines = page.get("lines", []) or []
        try:
            header_positions, _, _ = detect_column_positions(lines)
        except ValueError:
            header_positions = None

        found_for_page = None
        if header_positions is not None:
            found_for_page = find_totals_row_by_columns(lines, header_positions)

        if not found_for_page:
            for line in lines:
                line_text = line.get("line_text") or " ".join((w.get("text") or "") for w in line.get("words", []))
                if "total" not in (line_text or "").lower():
                    continue
                simple = _simple_totals_from_total_line(line)
                if simple:
                    found_for_page = simple
                    break

        if found_for_page and cur not in summaries:
            summaries[cur] = found_for_page
    return summaries

def build_currency_sections_balances(
    buckets: Dict[str, List[dict]],
    summaries: Dict[str, dict],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    all_currencies = set(buckets.keys()) | set(summaries.keys())
    for cur in sorted(all_currencies):
        txs = buckets.get(cur, [])

        tx_money_in  = round(sum(float(t["amount"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        tx_money_out = round(sum(float(t["amount"]) for t in txs if t.get("transaction_type") == "debit"), 2)

        s = summaries.get(cur, {}) or {}
        sum_opening = _num_or_none(s.get("opening"))
        sum_out     = _num_or_none(s.get("out"))
        sum_in      = _num_or_none(s.get("in"))
        sum_closing = _num_or_none(s.get("closing"))

        calculated = None
        if (sum_opening is not None) and (sum_in is not None) and (sum_out is not None):
            calculated = round(sum_opening + sum_in - sum_out, 2)

        out[cur] = {
            "balances": {
                "opening_balance": {"summary_table": sum_opening, "transactions_table": None},
                "money_in_total":  {"summary_table": sum_in,     "transactions_table": tx_money_in},
                "money_out_total": {"summary_table": sum_out,    "transactions_table": tx_money_out},
                "closing_balance": {"summary_table": sum_closing,"transactions_table": None, "calculated": calculated},
            },
            "transactions": txs,
        }
    return out

# ---------------------------------------------------------------------
# Transactions parser + running balance check
# ---------------------------------------------------------------------
def parse_transactions(pages: List[dict], iban: Optional[str] = None) -> List[dict]:
    def infer_page_currency(page: dict, prev_cur: Optional[str]) -> str:
        cur = page.get("currency")
        if isinstance(cur, str) and len(cur) == 3:
            return cur.upper()
        if iban:
            u = iban.upper()
            if u.startswith("GB"): return "GBP"
            if u.startswith("IE"): return "EUR"
        return prev_cur or "EUR"

    all_transactions: List[dict] = []
    prev_cur: Optional[str] = None
    seq = 0

    running_balance: Optional[float] = None
    have_seen_doc_balance = False

    for page in pages:
        page_currency = infer_page_currency(page, prev_cur)   # ← currency context for this page
        prev_cur = page_currency

        lines = page.get("lines", []) or []
        if not lines:
            continue

        try:
            header_positions, start_idx, end_idx = detect_column_positions(
                lines, require_balance=True
            )
        except ValueError:
            continue

        amount_cols = [v for k, v in header_positions.items() if k in ("in", "out") and v is not None]
        first_amount_x = min(amount_cols) if amount_cols else None

        for line in lines[start_idx + 1: end_idx]:
            df = pd.DataFrame(line.get("words", []))
            if df.empty:
                continue

            df.rename(columns={"word": "text", "x": "left"}, inplace=True, errors="ignore")
            df["line_text"] = line.get("line_text", "")
            df["amount"] = df["text"].apply(parse_currency)
            df["category"] = df.apply(
                lambda r: categorise_amount_by_left_edge(r.get("left", 0), header_positions, width=r.get("width")),
                axis=1
            )

            m = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", df["line_text"].iloc[0])
            if not m:
                continue
            transaction_date = parse_date(m.group(0))
            if not transaction_date:
                continue

            in_series  = df[df["category"] == "in"]["amount"]
            out_series = df[df["category"] == "out"]["amount"]
            bal_series = df[df["category"] == "bal"]["amount"]

            credit_amount = _num_or_none(in_series.iloc[0])  if not in_series.empty  else None
            debit_amount  = _num_or_none(out_series.iloc[0]) if not out_series.empty else None
            doc_balance   = _num_or_none(bal_series.iloc[0]) if not bal_series.empty else None

            # Exclude Pending/Reverted tables (no Balance column on row)
            if doc_balance is None:
                continue
            if credit_amount is None and debit_amount is None:
                continue

            clean_desc = transaction_description(line.get("words", []), header_positions.get("desc"), first_amount_x)

            credit = float(credit_amount or 0.0)
            debit  = float(debit_amount  or 0.0)
            delta  = round(credit - debit, 2)

            # (running balance checker unchanged; omitted for brevity)

            all_transactions.append({
                "seq": seq,
                "transaction_date": transaction_date,
                "transaction_type": "credit" if credit > 0 else "debit",
                "description": clean_desc,
                "amount": credit if credit > 0 else debit,
                "signed_amount": delta,
                "statement_balance": doc_balance,
                "currency": page_currency,      # ← NEW: tag the row with page currency
            })
            seq += 1

    return all_transactions

# ---------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------
def parse_statement(raw, client: str = "Unknown", account_type: str = "Unknown") -> dict:
    # IBAN from flat text (for bucket key only)
    full_text = "\n".join("\n".join(line.get("line_text", "") for line in page.get("lines", []))
                          for page in raw.get("pages", []))
    m = re.search(r"IBAN\s+([A-Z]{2}\d{2}[A-Z0-9]{11,30})", full_text)
    iban = m.group(1).strip() if m else None

    pages = raw.get("pages", []) or []

    # 1) parse rows
    transactions = parse_transactions(pages, iban=iban)

    # 2) totals from “Total a b c d”
    def _infer(page, prev):
        cur = page.get("currency")
        if isinstance(cur, str) and len(cur) == 3:
            return cur.upper()
        if iban:
            u = iban.upper()
            if u.startswith("GB"): return "GBP"
            if u.startswith("IE"): return "EUR"
        return prev or "EUR"

    summaries = collect_currency_summaries_from_totals(pages, _infer)

    # 3) balances section
    buckets = group_transactions_by_currency(transactions, iban)
    currencies = build_currency_sections_balances(buckets, summaries)

    # 4) dates from rows
    if transactions:
        all_dates = [t.get("transaction_date") for t in transactions if t.get("transaction_date")]
        start_date = min(all_dates) if all_dates else None
        end_date   = max(all_dates) if all_dates else None
    else:
        start_date = end_date = None

    # 5) id
    sid_basis = f"{raw.get('file_name') or ''}|{start_date or ''}|{end_date or ''}"
    statement_id = hashlib.sha1(sid_basis.encode("utf-8")).hexdigest()[:12] if sid_basis.strip("|") else None

    return {
        "statement_id": statement_id,
        "file_name": raw.get("file_name"),
        "institution": "Revolut",
        "account_type": account_type,
        "iban": iban,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
        "meta": (raw.get("meta") or {}),
    }
