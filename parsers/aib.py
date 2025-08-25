# aib.py
# AIB parser → multi-statement bundle (single statement for now)
# - Keeps transactions simple: {seq, transaction_date, transaction_type, description, amount}
# - Statement node contains: institution, account_type, account_holder, iban, bic,
#   statement_start_date, statement_end_date, currencies{...}
# - Top-level bundle: { schema_version, client, statements: [ <statement> ] }

from __future__ import annotations

import re
import hashlib
from typing import Optional, Dict, List, Any

import pandas as pd

from utils import parse_currency, parse_date, date_variants

# ----------------------------
# Regexes
# ----------------------------

RE_BAL_FWD   = re.compile(r"\bBALANCE\s*FORWARD\b", re.IGNORECASE)
RE_DATE_LONG = re.compile(r"\b(\d{1,2})\s+[A-Za-z]{3,9}\s+\d{4}\b")
RE_DATE_NUM  = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})\b")

# Accept either symbol or 3-letter code as a header currency token
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
    """
    Locate AIB headers "Date | Details | Debit €|£ | Credit €|£ | Balance €|£".
    Returns (header_positions, header_start_idx, len(lines)) or None.
    """
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
# Helpers
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
        if v is None and i + 1 < n and _is_currency_token((tokens[i + 1].get("text") or "").strip()):
            # Extend right edge to include the trailing currency token
            v = parse_currency(t + (tokens[i + 1].get("text") or ""), strip_currency=False)
            xr = _right_edge(tokens[i + 1])
        if v is None:
            continue
        if best is None or xr > best[0]:
            best = (xr, float(v))
    return (None, None) if best is None else best

# Strict "amount token" detector for description cleanup
AMOUNT_TOKEN_RE = re.compile(
    r"""
    ^[+\-]?                                  # optional sign
    (?:
        (?:\d{1,3}(?:[.,]\d{3})+|\d+)        # 1-3 + grouped thousands, or plain digits
    )
    [.,]\d{2}$                               # decimal sep + exactly two decimals
    """,
    re.VERBOSE,
)

def _looks_like_amount_token(tok: str) -> bool:
    """Strict check for standalone amount-looking tokens (for description cleanup only)."""
    if not tok:
        return False
    t = tok.strip()
    if not t:
        return False
    # allow single leading/trailing currency symbol
    if t[0] in {"€", "£"}:
        t = t[1:]
    elif t[-1:] in {"€", "£"}:
        t = t[:-1]
    return bool(AMOUNT_TOKEN_RE.match(t))


# ----------------------------
# Opening & Closing balances
# ----------------------------

def extract_opening_balance_and_start_date(pages: list[dict], debug: bool = False) -> tuple[float | None, str | None]:
    """Opening balance from the very first 'BALANCE FORWARD' line in the statement."""
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
            return opening_value, earliest_date

    return None, None


def extract_closing_balance(pages: list[dict]) -> Optional[float]:
    """Closing balance = last number near Balance €|£ column."""
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
# Transactions
# ----------------------------

def parse_transactions(pages: list[dict], iban: str | None = None) -> list[dict]:
    all_transactions: List[dict] = []
    seq = 0

    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_positions(lines)
        if not header:
            continue

        header_positions, start_idx, end_idx = header
        details_left  = int(header_positions["details_left"])
        debit_right   = int(header_positions["debit_right"])

        last_seen_date: Optional[str] = None

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

            # carry-forward date from this line text
            d = _find_date_in_text(ltxt)
            if d:
                last_seen_date = d
            if last_seen_date is None:
                continue

            df = pd.DataFrame(words)
            if df.empty:
                continue

            # geometry and categorisation
            df["right"] = df["left"] + df["width"]
            df["amount_val"] = df["text"].apply(lambda x: parse_currency(x, strip_currency=False))
            df["category"] = df.apply(lambda r: categorise_amount_by_right_edge(int(r["right"]), header_positions), axis=1)

            debit_vals  = df.loc[(df["amount_val"].notna()) & (df["category"] == "debit"),  ["right", "amount_val"]]
            credit_vals = df.loc[(df["amount_val"].notna()) & (df["category"] == "credit"), ["right", "amount_val"]]

            debit  = float(debit_vals.sort_values("right")["amount_val"].iloc[-1])  if not debit_vals.empty  else 0.0
            credit = float(credit_vals.sort_values("right")["amount_val"].iloc[-1]) if not credit_vals.empty else 0.0
            if debit == 0.0 and credit == 0.0:
                # likely a continuation / info line
                continue

            # --- Build description using the Details column span ---
            # Take tokens whose *center* lies between Details-left and Debit-right
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

                # remove the known date (any printed variant) from the parts
                if last_seen_date and parts:
                    variants = set(date_variants(last_seen_date))
                    parts = [p for p in parts if p not in variants]
                clean_desc = " ".join(parts).strip(" -|")
            else:
                # Fallback: start from line_text and strip only strict amount tokens + currency + the date
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
                "transaction_date": last_seen_date,
                "transaction_type": "credit" if credit > 0 else "debit",
                "description": clean_desc,
                "amount": credit if credit > 0 else debit,  # always positive
                "signed_amount": (credit if credit > 0 else -debit),  # + for credit, - for debit
            })

            seq += 1

    return all_transactions

# ----------------------------
# IBAN extraction & currency inference
# ----------------------------

def extract_iban(pages: list[dict]) -> str | None:
    """
    Extract IBAN for AIB statements.
    Assumes format like: "IBAN: GB03 FTBK 9383 7828 8330 49   (BIC: FTBKGB2B)"
    Returns compact IBAN string with no spaces.
    """
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
    """
    If IBAN is missing, attempt to infer currency by inspecting header currency tokens.
    """
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

    # IBAN & currency detection
    iban = extract_iban(pages)
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    if not iban:
        inferred = infer_currency_from_headers(pages)
        if inferred:
            currency = inferred

    # Transactions (flat)
    transactions = parse_transactions(pages, iban=iban)

    # Opening and closing (from table)
    opening_val, start_date = extract_opening_balance_and_start_date(pages)
    closing_val = extract_closing_balance(pages)

    # Totals from transactions
    money_in  = round(sum(t["amount"] for t in transactions if t["transaction_type"] == "credit"), 2)
    money_out = round(sum(t["amount"] for t in transactions if t["transaction_type"] == "debit"), 2)

    closing_calc = None
    if opening_val is not None:
        closing_calc = round(opening_val + money_in - money_out, 2)

    # Statement date range
    if transactions:
        all_dates = [t["transaction_date"] for t in transactions if t["transaction_date"]]
        start_date = start_date or (min(all_dates) if all_dates else None)
        end_date   = max(all_dates) if all_dates else None
    else:
        end_date = None

    currencies = {
        currency: {
            "balances": {
                "opening_balance": {
                    "summary_table": None,                # AIB has no summary table
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

    # Optional lightweight statement_id
    sid_basis = f"{raw_ocr.get('file_name') or ''}|{start_date or ''}|{end_date or ''}"
    statement_id = hashlib.sha1(sid_basis.encode("utf-8")).hexdigest()[:12] if sid_basis.strip("|") else None

    return {
        "statement_id": statement_id,
        "file_name": raw_ocr.get("file_name"),
        "institution": "AIB",
        "account_type": account_type,
        "iban": iban,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }

# ----------------------------
# Public entrypoint: bundle with one statement
# ----------------------------

def parse_statement(raw_ocr: dict, client: str = "Unknown", account_type: str = "Unknown", debug: bool = True) -> dict:
    """
    Kept for compatibility. Returns ONLY the inner statement object (not the bundle).
    Prefer 'parse_statement_bundle' going forward.
    """
    return _make_statement_node(raw_ocr, client=client, account_type=account_type, debug=debug)
