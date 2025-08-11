import re
import math
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any
import pandas as pd
from utils import parse_currency, parse_date
from lang import HEADER_SYNONYMS  # expects keys: date, description, money_out, money_in, balance


def detect_column_positions(lines: List[dict]) -> tuple[dict, int, int]:
    """
    Find the transaction header row (Date / Description / Money out / Money in / Balance),
    supporting synonyms in multiple languages via HEADER_SYNONYMS.
    Returns (header_positions, header_idx, end_idx=len(lines)).
    header_positions: {"date": int|None, "desc": int|None, "out": int|None, "in": int|None, "bal": int|None}
    """

    def match_header(text: str, keywords: List[str]) -> bool:
        s = text.lower()
        return any(k in s for k in keywords)

    for idx, line in enumerate(lines):
        line_text = " ".join(w.get("text", "") for w in line.get("words", []))
        if (
            match_header(line_text, HEADER_SYNONYMS["date"]) and
            match_header(line_text, HEADER_SYNONYMS["description"]) and
            (match_header(line_text, HEADER_SYNONYMS["money_out"]) or
             match_header(line_text, HEADER_SYNONYMS["money_in"]))
        ):
            positions: Dict[str, Optional[int]] = {
                "date": None, "description": None, "money_out": None, "money_in": None, "balance": None
            }

            words = line.get("words", [])
            for i, word in enumerate(words):
                text = (word.get("text") or "").lower()
                x = int(word.get("left", word.get("x", 0)))

                if any(text == k or k in text for k in HEADER_SYNONYMS["date"]):
                    positions["date"] = x
                elif any(text == k or k in text for k in HEADER_SYNONYMS["description"]):
                    positions["description"] = x
                elif any(text == k or k in text for k in HEADER_SYNONYMS["balance"]):
                    positions["balance"] = x

                if i + 1 < len(words):
                    combined = f"{text} {words[i+1].get('text','').lower()}"
                    if any(combined == k for k in HEADER_SYNONYMS["money_out"]):
                        positions["money_out"] = x
                    elif any(combined == k for k in HEADER_SYNONYMS["money_in"]):
                        positions["money_in"] = x

            return ({
                "date": positions["date"],
                "desc": positions["description"],
                "out":  positions["money_out"],
                "in":   positions["money_in"],
                "bal":  positions["balance"],
            }, idx, len(lines))

    raise ValueError("❌ Could not find transaction header row.")


def categorise_amount_by_left_edge(x1, header_positions, margin=60):
    """
    Classify an amount token into 'out' | 'in' | 'bal' by proximity to column left edges.
    """
    x1 = int(x1)
    best = ("unknown", margin + 1)
    for k in ("out", "in", "bal"):
        pos = header_positions.get(k)
        if pos is None:
            continue
        d = abs(x1 - int(pos))
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
    return text and "total" in text.lower()  # works for EN/ES/FR OCR


def _amount_tokens_from_line(line: dict) -> List[Tuple[int, float]]:
    """
    Extract (x_left, value) for each token that parses as a number.
    """
    out: List[Tuple[int, float]] = []
    for w in line.get("words", []):
        t = w.get("text", "") or ""
        val = parse_currency(t, strip_currency=True)
        if val is None:
            continue
        x_left = int(w.get("left", w.get("x", 0)))
        out.append((x_left, float(val)))
    out.sort(key=lambda p: p[0])  # left-to-right
    return out


def _classify_total_amount_x(x_left: int, header_positions: dict, margin=60) -> Optional[str]:
    """
    Map an amount x position to 'out' | 'in' | 'closing' by nearest column (Money out / Money in / Balance).
    """
    x_left = int(x_left)
    candidates = {
        "out": header_positions.get("out"),
        "in": header_positions.get("in"),
        "closing": header_positions.get("bal"),
    }
    best = (None, margin + 1)
    for k, pos in candidates.items():
        if pos is None:
            continue
        d = abs(x_left - int(pos))
        if d < best[1]:
            best = (k, d)
    return best[0] if best[0] is not None and best[1] <= margin else None


def find_totals_row_by_columns(lines: List[dict], header_positions: dict) -> Optional[dict]:
    """
    Find a 'Total …' line. Map amounts by nearest column:
      b -> Money out   | c -> Money in   | d -> Balance
    Then compute a = d - c + b.
    Returns {'opening': float|None, 'out': float|None, 'in': float|None, 'closing': float|None}
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

        amounts = _amount_tokens_from_line(line)  # -> [(x_left, value), ...]
        if not amounts:
            continue

        # pick b/c/d by nearest header x
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

        totals = {
            "opening": opening,
            "out": money_out,
            "in":  money_in,
            "closing": closing,
        }

        if any(v is not None for v in totals.values()):
            return totals

    return None


def statement_date_range(pages: List[dict]) -> tuple[Optional[str], Optional[str]]:
    """
    Optional: pull "from X to Y" date range if present in header text.
    """
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


def group_transactions_by_currency(transactions: List[dict]) -> Dict[str, List[dict]]:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for tx in transactions:
        cur = tx.get("amount", {}).get("currency")
        if cur:
            buckets[cur].append(tx)
    for cur in buckets:
        buckets[cur].sort(key=lambda t: (t.get("transactions_date") or "",
                                         t.get("balance_after_statement", {}).get("value", 0.0)))
    return buckets


def _simple_totals_from_total_line(line: dict) -> dict | None:
    """
    Works even if there is no transactions header on the page.
    For a line that contains 'Total', read the four amounts left→right.
    Returns {'opening','out','in','closing'} or None.
    """
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

    amts.sort(key=lambda t: t[0])  # left→right
    vals = [v for _, v in amts]

    # take the first four numeric cells on that line as a,b,c,d
    a, b, c, d = vals[0], vals[1], vals[2], vals[3]

    # for robustness, recompute opening from the other three
    a = round(float(d) - float(c) + float(b), 2)

    return {"opening": a, "out": b, "in": c, "closing": d}


def detect_summary_column_positions(lines: List[dict]) -> Optional[dict]:
    """
    Detect left x-positions of Money out / Money in / Closing balance
    from the Balance summary header (no transactions table on the page).
    Returns {"out": int|None, "in": int|None, "bal": int|None} or None.
    Keeps it simple (English); extend with synonyms if needed.
    """
    def has_any(hay, needles):
        s = hay.lower()
        return all(n in s for n in needles)

    for line in lines:
        line_text = " ".join((w.get("text") or "") for w in line.get("words", []))
        if not line_text:
            continue

        # Balance summary header usually contains these labels
        if (
            ("opening" in line_text.lower() and "balance" in line_text.lower()) and
            ("money" in line_text.lower()) and
            ("closing" in line_text.lower() and "balance" in line_text.lower())
        ):
            pos = {"out": None, "in": None, "bal": None}
            words = line.get("words", [])

            for i, w in enumerate(words):
                t = (w.get("text") or "").lower()
                x = int(w.get("left", w.get("x", 0)))

                # two-word matches for money out / money in
                if i + 1 < len(words):
                    nxt = (words[i+1].get("text") or "").lower()
                    pair = f"{t} {nxt}"
                    if pair == "money out":
                        pos["out"] = x
                    elif pair == "money in":
                        pos["in"] = x

                # single-word closing / balance → take the 'closing' word
                if t == "closing":
                    pos["bal"] = x

            # if we at least found one of out/in/bal, return what we have
            if any(v is not None for v in pos.values()):
                return pos

    return None


def collect_currency_summaries_from_totals(pages: list[dict], infer_page_currency) -> dict[str, dict]:
    """
    Find 'Total a b c d' for each page’s currency, even if no transactions table exists.
    Tries column-aligned method first; falls back to simple left→right four-amounts.
    """
    summaries: dict[str, dict] = {}
    prev_cur: str | None = None

    for page in pages:
        cur = infer_page_currency(page, prev_cur)
        prev_cur = cur

        lines = page.get("lines", []) or []
        header_positions = None
        try:
            header_positions, _, _ = detect_column_positions(lines)
        except ValueError:
            header_positions = None

        found_for_page = None

        # Pass 1: column-aligned (if we have header positions)
        if header_positions is not None:
            found_for_page = find_totals_row_by_columns(lines, header_positions)

        # Pass 2: simple fallback — read four amounts on the 'Total' line
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


def build_currency_sections(buckets: Dict[str, List[dict]], summaries: Dict[str, dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    all_currencies = set(buckets.keys()) | set(summaries.keys())

    for cur in sorted(all_currencies):
        txs = buckets.get(cur, [])

        # derivations from transactions
        money_in  = sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit")
        money_out = sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit")

        # infer opening/closing from txs if needed
        opening_from_txs = None
        if txs:
            first = txs[0]
            bal = first.get("balance_after_statement", {}).get("value")
            amt = first.get("amount", {}).get("value")
            typ = first.get("transaction_type")
            if bal is not None and amt is not None and typ in ("credit", "debit"):
                opening_from_txs = round(float(bal) - float(amt), 2) if typ == "credit" else round(float(bal) + float(amt), 2)
        closing_stmt_from_txs = txs[-1].get("balance_after_statement", {}).get("value") if txs else None

        # overlay statement totals if present
        s = summaries.get(cur, {}) or {}
        opening = s.get("opening") if s.get("opening") is not None else opening_from_txs
        total_out = s.get("out") if s.get("out") is not None else round(money_out, 2)
        total_in  = s.get("in")  if s.get("in")  is not None else round(money_in, 2)
        closing_stmt = s.get("closing") if s.get("closing") is not None else closing_stmt_from_txs

        closing_calc = None
        if opening is not None:
            closing_calc = round(float(opening) + float(total_in) - float(total_out), 2)

        out[cur] = {
            "opening_balance": None if opening is None else {"value": float(opening), "currency": cur},
            "money_in_total":  {"value": float(total_in),  "currency": cur},
            "money_out_total": {"value": float(total_out), "currency": cur},
            "closing_balance_statement": None if closing_stmt is None else {"value": float(closing_stmt), "currency": cur},
            "closing_balance_calculated": None if closing_calc is None else {"value": float(closing_calc), "currency": cur},
            "transactions": txs,
        }
    return out


def parse_transactions(pages: List[dict], iban: Optional[str] = None) -> List[dict]:
    """
    Parse all transaction rows across pages. Returns a flat list of transaction dicts.
    Each includes its own currency in the 'amount' block.
    """

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
    running_balance_cents: Optional[int] = None
    current_currency: Optional[str] = None

    # (Optional) pick up header date range; not essential for parsing
    start_date_str, _ = statement_date_range(pages)
    initial_value_date = parse_date(start_date_str) if start_date_str else None
    last_seen_date = initial_value_date

    prev_cur: Optional[str] = None

    totals_in: dict[str, float] = {}
    totals_out: dict[str, float] = {}

    seq = 0  # monotonic sequence to preserve statement order

    for page_num, page in enumerate(pages):
        page_currency = infer_page_currency(page, prev_cur)
        prev_cur = page_currency

        if current_currency != page_currency:
            running_balance_cents = None
            current_currency = page_currency

        lines = page.get("lines", [])
        if not lines:
            continue

        try:
            header_positions, start_idx, end_idx = detect_column_positions(lines)
        except ValueError:
            continue

        # For description slicing
        amount_cols = [v for k, v in header_positions.items() if k in ("in", "out", "bal") and v is not None]
        first_amount_x = min(amount_cols) if amount_cols else None

        # Iterate rows below header
        for line in lines[start_idx + 1: end_idx]:
            df = pd.DataFrame(line.get("words", []))
            if df.empty:
                continue

            df.rename(columns={"word": "text", "x": "left"}, inplace=True, errors="ignore")
            df["line_text"] = line.get("line_text", "")

            df["amount"] = df["text"].apply(parse_currency)
            df["category"] = df["left"].apply(lambda x: categorise_amount_by_left_edge(x, header_positions))

            # Require a date on the line
            date_match = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", df["line_text"].iloc[0])
            if not date_match:
                continue
            parsed_date = parse_date(date_match.group(0))
            if not parsed_date:
                continue
            transaction_date = parsed_date
            last_seen_date = transaction_date

            # Amounts by category
            in_series = df[df["category"] == "in"]["amount"]
            out_series = df[df["category"] == "out"]["amount"]
            bal_series = df[df["category"] == "bal"]["amount"]

            credit_amount = in_series.iloc[0] if not in_series.empty else None
            debit_amount = out_series.iloc[0] if not out_series.empty else None
            balance = bal_series.iloc[0] if not bal_series.empty else None

            # --- NEW: normalize NaNs to None
            credit_amount = _num_or_none(credit_amount)
            debit_amount = _num_or_none(debit_amount)
            balance = _num_or_none(balance)

            # Must have date, balance, and either in or out
            if balance is None or (credit_amount is None and debit_amount is None):
                continue

            clean_desc = transaction_description(
                line.get("words", []),
                header_positions.get("desc"),
                first_amount_x
            )

            credit_amount = credit_amount or 0.0
            debit_amount = debit_amount or 0.0

            # --- accumulate per-currency totals
            totals_in[current_currency] = totals_in.get(current_currency, 0.0) + credit_amount
            totals_out[current_currency] = totals_out.get(current_currency, 0.0) + debit_amount

            credit_c = int(round(credit_amount * 100))
            debit_c = int(round(debit_amount * 100))

            if running_balance_cents is None:
                # balance is guaranteed not None now
                running_balance_cents = int(round(balance * 100))
            else:
                running_balance_cents += credit_c - debit_c

            calculated_balance = round(running_balance_cents / 100.0, 2)

            if not math.isclose(calculated_balance, float(balance), abs_tol=0.01):
                discrepancy = round(float(balance) - calculated_balance, 2)
                print("\n⚠️ BALANCE DISCREPANCY DETECTED")
                print(f"📄 Page: {page_num + 1} | 💱 Currency: {current_currency}")
                print(f"📆 Date: {transaction_date}")
                print(f"📝 Description: {clean_desc}")
                print(f"💸 Credit: {credit_amount:.2f} | Debit: {debit_amount:.2f}")
                print(f"📊 Calculated Balance: {calculated_balance:.2f}")
                print(f"📄 Statement Balance:  {float(balance):.2f}")
                print(f"🧮 Discrepancy: {discrepancy:+.2f}")
                print(f"🔤 OCR Line: {df['line_text'].iloc[0]}")

            amt_value = credit_amount if credit_amount > 0 else debit_amount
            all_transactions.append({
                "transactions_date": transaction_date,
                "transaction_type": "credit" if credit_amount > 0 else "debit",
                "description": clean_desc,
                "amount": {
                    "value": amt_value,
                    "currency": current_currency,
                },
                "balance_after_statement": {
                    "value": float(balance),
                    "currency": current_currency,
                },
                "balance_after_calculated": {
                    "value": calculated_balance,
                    "currency": current_currency,
                },
            })

    return all_transactions

def debug_print_currency_summary(currencies: dict[str, dict]) -> None:
    total_tx = sum(len(sec.get("transactions", [])) for sec in currencies.values())
    print(f"\n📄 TOTAL REVOLUT TRANSACTIONS: {total_tx}")
    for cur in sorted(currencies.keys()):
        mi = currencies[cur]["money_in_total"]["value"]
        mo = currencies[cur]["money_out_total"]["value"]
        print(f"  • {cur}: IN {mi:.2f} | OUT {mo:.2f}")


def parse_statement(raw_ocr, client: str = "Unknown", account_type: str = "Unknown"):
    # Flat text for IBAN extraction
    full_text = "\n".join("\n".join(line.get("line_text", "") for line in page.get("lines", []))
                          for page in raw_ocr.get("pages", []))

    iban_match = re.search(r"IBAN\s+([A-Z]{2}\d{2}[A-Z0-9]{11,30})", full_text)
    iban = iban_match.group(1).strip() if iban_match else None

    pages = raw_ocr.get("pages", [])

    # 1) parse transactions
    transactions = parse_transactions(pages, iban=iban)

    # 2) infer page currency logic for summaries
    def _infer(page, prev):
        cur = page.get("currency")
        if isinstance(cur, str) and len(cur) == 3:
            return cur.upper()
        if iban:
            u = iban.upper()
            if u.startswith("GB"): return "GBP"
            if u.startswith("IE"): return "EUR"
        return prev or "EUR"

    # 3) collect totals from "Total a b c d" rows aligned by columns
    summaries = collect_currency_summaries_from_totals(pages, _infer)

    # 4) group and build per-currency sections
    buckets = group_transactions_by_currency(transactions)
    currencies = build_currency_sections(buckets, summaries)
    debug_print_currency_summary(currencies)

    # 5) global dates from transactions (language-agnostic)
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date = min(all_dates) if all_dates else None
        end_date = max(all_dates) if all_dates else None
    else:
        start_date = end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,
        "institution": "Revolut",
        "account_type": account_type,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies
    }
