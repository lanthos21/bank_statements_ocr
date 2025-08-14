# aib_current.py
# AIB parser:
# - Header: "Date | Details | Debit € | Credit € | Balance €"
# - Amounts are RIGHT-JUSTIFIED under the three amount columns
# - Robust opening balance ("BALANCE FORWARD …") and closing balance (last Balance €)
# - Transactions shaped like Revolut/BOI JSON

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

from utils import parse_currency, parse_date


# ----------------------------
# Regexes
# ----------------------------

RE_BAL_FWD = re.compile(r"\bBALANCE\s*FORWARD\b", re.IGNORECASE)

# dates like "26 Jan 2024"
RE_DATE_LONG = re.compile(r"\b(\d{1,2})\s+[A-Za-z]{3,9}\s+\d{4}\b")
# dates like "26/01/2024" or "26-01-2024"
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

    We store RIGHT edges for debit/credit/balance (amounts are right-justified).
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
                # prefer the trailing '€' token if present
                if i + 1 < n and (words[i + 1].get("text") or "").strip() == "€":
                    pos["debit_right"] = right_edge(words[i + 1])
                else:
                    pos["debit_right"] = right_edge(w)

            elif t == "credit" and pos["credit_right"] is None:
                if i + 1 < n and (words[i + 1].get("text") or "").strip() == "€":
                    pos["credit_right"] = right_edge(words[i + 1])
                else:
                    pos["credit_right"] = right_edge(w)

            elif t == "balance" and pos["balance_right"] is None:
                # AIB usually has "Balance €" with '€' as a separate token
                if i + 1 < n and (words[i + 1].get("text") or "").strip() == "€":
                    pos["balance_right"] = right_edge(words[i + 1])
                else:
                    pos["balance_right"] = right_edge(w)

        if all(pos[k] is not None for k in ("date_left", "details_left", "debit_right", "credit_right", "balance_right")):
            return pos, idx, len(lines)

    return None


def categorise_amount_by_right_edge(x_right: int, header_positions: dict, margin: int = 120) -> str:
    """
    Classify token into 'debit' | 'credit' | 'balance' by proximity to the RIGHT edges.
    Uses a generous margin so very wide numbers still land in-window.
    """
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
    """
    Return (x_right, value) for the numeric whose RIGHT edge is farthest right on this line.
    Tolerant to a trailing '€' as a separate token.
    """
    best = None  # (x_right, value)
    tokens = words or []
    n = len(tokens)

    for i, w in enumerate(tokens):
        t = (w.get("text") or "").strip()
        if not t:
            continue

        v = parse_currency(t, strip_currency=False)
        xr = _right_edge(w)

        # join with a trailing '€' token if present to extend visual edge
        if v is None and i + 1 < n:
            nxt = (tokens[i + 1].get("text") or "").strip()
            if nxt == "€":
                v = parse_currency(t + nxt, strip_currency=False)
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
    Original logic with detailed debug tracing:
    - On each 'BALANCE FORWARD' line, pick the rightmost numeric on that line.
    - If none, peek the next line and take its rightmost numeric.
    - Keep the earliest date seen alongside a 'BALANCE FORWARD'.
    """
    def dbg(*a):
        if debug:
            print(*a)

    def _num_candidates(words: list[dict]) -> list[dict]:
        """
        Return a list of numeric candidates with token text, parsed value, and geometry.
        Joins a trailing '€' token to extend the right edge if present.
        """
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
                        # extend right edge to include '€'
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

    earliest_date: Optional[str] = None
    opening_value: Optional[float] = None

    for pidx, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        i = 0
        while i < len(lines):
            line = lines[i]
            lt = (line.get("line_text") or "")
            if not RE_BAL_FWD.search(lt):
                i += 1
                continue

            dbg(f"\n🔎 Page {pidx}, line {i} contains 'BALANCE FORWARD':")
            dbg(f"    line_text: {lt!r}")

            # 1) same-line candidates (your current approach)
            words = line.get("words", []) or []
            cands = _num_candidates(words)

            if cands:
                # choose the rightmost numeric (current behavior)
                cands_sorted = sorted(cands, key=lambda d: d["right"])
                chosen = cands_sorted[-1]
                dbg("    • Same-line numeric candidates (rightmost wins):")
                for c in cands_sorted:
                    dbg(f"      - token={c['text']!r:>12}  val={c['value']:>12,.2f}  left={c['left']:>5}  right={c['right']:>5}")
                dbg(f"    ✅ Chosen (same line): val={chosen['value']:.2f}  right={chosen['right']}")

                val = float(chosen["value"])

            else:
                dbg("    • No numeric on same line → peeking next line for rightmost numeric…")
                val = None
                if i + 1 < len(lines):
                    next_words = lines[i + 1].get("words", []) or []
                    next_cands = _num_candidates(next_words)
                    if next_cands:
                        next_sorted = sorted(next_cands, key=lambda d: d["right"])
                        chosen = next_sorted[-1]
                        for c in next_sorted:
                            dbg(f"      - (next) token={c['text']!r:>12}  val={c['value']:>12,.2f}  left={c['left']:>5}  right={c['right']:>5}")
                        dbg(f"    ✅ Chosen (next line): val={chosen['value']:.2f}  right={chosen['right']}")
                        val = float(chosen["value"])
                    else:
                        dbg("    ⚠️ No numeric candidates found on next line either.")

            # 2) date on the BALANCE FORWARD line (if present)
            d = _find_date_in_text(lt)
            if d:
                dbg(f"    📅 Date detected on line: {d}")

            # 3) apply selection to earliest
            if val is not None:
                if d:
                    if earliest_date is None or d < earliest_date:
                        dbg(f"    ⬆️ Setting earliest_date={d}, opening_value={val:.2f}")
                        earliest_date = d
                        opening_value = val
                    else:
                        dbg(f"    ℹ️ Found later date {d}; keeping earliest_date={earliest_date}, opening={opening_value}")
                elif opening_value is None:
                    dbg(f"    ⬆️ Setting opening_value={val:.2f} (no date on line)")
                    opening_value = val
            else:
                dbg("    ❌ No opening value resolved for this occurrence.")

            i += 1

    dbg("\n===== OPENING BALANCE SUMMARY =====")
    dbg(f"opening_value={opening_value}  earliest_date={earliest_date}\n")
    return opening_value, earliest_date


def extract_closing_balance(pages: list[dict]) -> Optional[float]:
    """
    Closing balance = the last number whose RIGHT edge sits near the Balance € column
    on any page below a detected header.
    """
    best: Optional[float] = None

    # Pre-compute headers per page
    headers: dict[int, dict] = {}
    for pidx, page in enumerate(pages or []):
        hdr = detect_column_positions(page.get("lines", []) or [])
        headers[pidx] = hdr[0] if hdr else None

    for pidx, page in enumerate(pages or []):
        pos = headers.get(pidx)
        if not pos:
            continue

        bal_r = int(pos["balance_right"])
        margin = 120

        lines = page.get("lines", []) or []
        for line in lines:
            words = line.get("words", []) or []
            if not words:
                continue
            xr, val = _rightmost_numeric_with_x(words)
            if val is None or xr is None:
                continue
            # accept only numbers that visually end near the Balance column's right edge
            if abs(int(xr) - bal_r) <= margin:
                best = float(val)  # last wins, pages are in order

    return best

# ----------------------------
# Transactions (AIB)
# ----------------------------

def parse_transactions(
    pages: list[dict],
    iban: str | None = None
) -> list[dict]:
    """
    Parse all AIB transaction rows across pages. Returns a flat list of transactions.
    Numbers are classified by RIGHT edge vs header's Debit/Credit/Balance positions.
    """
    all_transactions: List[dict] = []
    current_currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    seq = 0

    for page_num, page in enumerate(pages or []):
        lines = page.get("lines", []) or []
        header = detect_column_positions(lines)
        if not header:
            continue

        header_positions, start_idx, end_idx = header

        last_seen_date: Optional[str] = None

        for line in lines[start_idx + 1: end_idx]:
            words = line.get("words", []) or []
            if not words:
                continue

            # Skip the "BALANCE FORWARD" transaction as a row; we record it in metadata
            ltxt = line.get("line_text", "") or ""
            if RE_BAL_FWD.search(ltxt):
                # still update last_seen_date
                d = _find_date_in_text(ltxt)
                if d:
                    last_seen_date = d
                continue

            df = pd.DataFrame(words)
            if df.empty:
                continue

            df["right"] = df["left"] + df["width"]
            df["line_text"] = ltxt

            # carry-forward date
            d = _find_date_in_text(ltxt)
            if d:
                last_seen_date = d
            transaction_date = last_seen_date
            if transaction_date is None:
                # until we meet the first row that has a date
                continue

            # classify numeric tokens by RIGHT edge proximity to each amount column
            # (Only numeric-looking tokens are considered)
            def _val(x):
                return parse_currency(x, strip_currency=False)

            df["amount_val"] = df["text"].apply(_val)
            df["category"]   = df.apply(
                lambda r: categorise_amount_by_right_edge(int(r["right"]), header_positions), axis=1
            )

            # Get one value per category (prefer the rightmost numeric on the line)
            debit_vals   = df.loc[(df["amount_val"].notna()) & (df["category"] == "debit"),   ["right", "amount_val"]]
            credit_vals  = df.loc[(df["amount_val"].notna()) & (df["category"] == "credit"),  ["right", "amount_val"]]
            balance_vals = df.loc[(df["amount_val"].notna()) & (df["category"] == "balance"), ["right", "amount_val"]]

            debit  = float(debit_vals.sort_values("right")["amount_val"].iloc[-1])   if not debit_vals.empty   else 0.0
            credit = float(credit_vals.sort_values("right")["amount_val"].iloc[-1])  if not credit_vals.empty  else 0.0
            stmt_balance = float(balance_vals.sort_values("right")["amount_val"].iloc[-1]) if not balance_vals.empty else None

            if debit == 0.0 and credit == 0.0 and stmt_balance is None:
                # "details only" continuation line (no amounts) — skip
                continue

            # Description: remove the date token to keep the merchant text clean
            clean_desc = ltxt
            dd = RE_DATE_LONG.search(clean_desc) or RE_DATE_NUM.search(clean_desc)
            if dd:
                clean_desc = clean_desc.replace(dd.group(0), "").strip(" |-")

            all_transactions.append({
                "seq": seq,
                "transactions_date": transaction_date,
                "transaction_type": "credit" if credit > 0 else "debit",
                "description": clean_desc,
                "amount": {
                    "value": credit if credit > 0 else debit,
                    "currency": current_currency,
                },
                "balance_after_statement": None if stmt_balance is None else {
                    "value": stmt_balance,
                    "currency": current_currency,
                },
                "balance_after_calculated": None,
            })
            seq += 1

    return all_transactions


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
        buckets[cur].sort(key=lambda t: t.get("seq", 0))  # preserve statement order
    return buckets


def _derive_open_close_from_rows(txs: List[dict]) -> tuple[float | None, float | None]:
    """
    Opening from rows (fallback): reverse the first transaction against its statement balance (if present).
    Closing from rows: last statement balance (if present).
    """
    if not txs:
        return None, None

    first = txs[0]
    last  = txs[-1]

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
# Public entrypoint (AIB)
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


def parse_statement(raw_ocr, client="Unknown", account_type="Unknown", debug: bool = True):
    def dbg(*a):
        if debug:
            print(*a)

    pages = raw_ocr.get("pages", []) or []

    # IBAN & currency guess
    full_text = "\n".join("\n".join(line.get("line_text", "") for line in page.get("lines", [])) for page in pages)
    iban = extract_iban(pages)
    currency_hint = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    # Transactions (flat)
    transactions = parse_transactions(pages, iban=iban)
    dbg(f"🧾 Parsed {len(transactions)} transactions")

    # Group by currency (likely one)
    buckets = _group_by_currency(transactions)
    dbg("🔑 Buckets from rows:", list(buckets.keys()))

    # Build per-currency sections from rows
    currencies = _build_currency_sections_from_rows(buckets)
    for k, sec0 in currencies.items():
        dbg(f"   - {repr(k)} BEFORE overlay: opening_from_rows={sec0.get('opening_balance')} "
            f"in={sec0.get('money_in_total')} out={sec0.get('money_out_total')}")

    # Explicit scans
    opening_val, start_date = extract_opening_balance_and_start_date(pages)
    closing_val = extract_closing_balance(pages)
    dbg(f"📘 Explicit opening_val={opening_val} start_date={start_date}")
    dbg(f"📘 Explicit closing_val={closing_val}")

    # Choose target currency robustly:
    # 1) Prefer EUR if present; 2) else first bucket; 3) else currency_hint
    if "EUR" in currencies:
        target_cur = "EUR"
    else:
        target_cur = next(iter(currencies.keys()), currency_hint)
    dbg("🎯 target_cur for overlay:", repr(target_cur))

    # Ensure target bucket exists if no tx were parsed (edge case)
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

    # Overlay explicit opening / closing
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

    # Recompute a calculated closing if we now have opening + totals
    if sec.get("opening_balance") and sec.get("money_in_total") and sec.get("money_out_total"):
        o = float(sec["opening_balance"]["value"])
        inm = float(sec["money_in_total"]["value"])
        outm = float(sec["money_out_total"]["value"])
        sec["closing_balance_calculated"] = {"value": round(o + inm - outm, 2), "currency": target_cur}
        dbg("🧮 closing_balance_calculated:", sec["closing_balance_calculated"])

    # Final per-currency snapshot
    for k, s in currencies.items():
        dbg(f"   - {repr(k)} AFTER overlay: opening={s.get('opening_balance')} "
            f"closing_stmt={s.get('closing_balance_statement')} closing_calc={s.get('closing_balance_calculated')}")

    # Statement dates from transactions (fallback to opening date if none)
    if transactions:
        all_dates = [t.get("transactions_date") for t in transactions if t.get("transactions_date")]
        start_date_tx = min(all_dates) if all_dates else None
        end_date_tx   = max(all_dates) if all_dates else None
        start_date = start_date or start_date_tx
        end_date   = end_date_tx
    else:
        end_date = None

    return {
        "client": client,
        "file_name": raw_ocr.get("file_name"),
        "account_holder": None,
        "institution": "AIB",
        "account_type": account_type,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "currencies": currencies,
    }
