# validator.py

import math

ABS_TOL = 0.01  # cents tolerance


def _num(x):
    return None if x is None else float(x)


def _val(node):
    return None if not node else _num(node.get("value"))


def _fmt(x):
    return "None" if x is None else f"{float(x):.2f}"


def _cmp(a, b):
    """
    Compare with tolerance; return (delta = b - a, ok)
    If either is None, returns (None, None).
    """
    if a is None or b is None:
        return None, None
    d = round(float(b) - float(a), 2)
    return d, math.isclose(float(a), float(b), abs_tol=ABS_TOL)


def _opening_from_rows(txs):
    """
    Derive opening using only the first transaction row:
    opening = first_stmt_balance - amount (credit) OR + amount (debit).
    """
    if not txs:
        return None
    first = txs[0]
    bal = _val(first.get("balance_after_statement"))
    amt = _num(first.get("amount", {}).get("value"))
    typ = first.get("transaction_type")
    if bal is None or amt is None or typ not in ("credit", "debit"):
        return None
    return round(bal - amt, 2) if typ == "credit" else round(bal + amt, 2)


def _closing_from_rows(txs):
    """
    Last statement balance in the rows, if present.
    """
    if not txs:
        return None
    last = txs[-1]
    return _val(last.get("balance_after_statement"))


def validate_statement_json(data: dict) -> dict:
    import math

    ABS_TOL = 0.01

    def _num(x):
        return None if x is None else float(x)

    def _val(node):
        return None if not node else _num(node.get("value"))

    def _fmt(x):
        return "None" if x is None else f"{float(x):.2f}"

    def _cmp(a, b):
        if a is None or b is None:
            return None, None
        d = round(float(b) - float(a), 2)
        return d, math.isclose(float(a), float(b), abs_tol=ABS_TOL)

    def _opening_from_rows(txs):
        if not txs:
            return None
        first = txs[0]
        bal = _val(first.get("balance_after_statement"))
        amt = _num(first.get("amount", {}).get("value"))
        typ = first.get("transaction_type")
        if bal is None or amt is None or typ not in ("credit", "debit"):
            return None
        return round(bal - amt, 2) if typ == "credit" else round(bal + amt, 2)

    def _closing_from_rows(txs):
        if not txs:
            return None
        last = txs[-1]
        return _val(last.get("balance_after_statement"))

    print(f"\n=== VALIDATION for: {data.get('file_name')} ===")
    report = {"file_name": data.get("file_name"), "currencies": {}, "ok": True}
    currencies = data.get("currencies", {}) or {}

    for cur in sorted(currencies.keys()):
        sec = currencies[cur]
        txs = sec.get("transactions", []) or []
        n = len(txs)

        opening_s = _val(sec.get("opening_balance"))
        in_s      = _val(sec.get("money_in_total"))
        out_s     = _val(sec.get("money_out_total"))
        closing_s = _val(sec.get("closing_balance_statement"))

        in_tx  = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit"), 2)
        out_tx = round(sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit"), 2)

        opening_from_summary = (
            None if (closing_s is None or in_s is None or out_s is None)
            else round(closing_s - in_s + out_s, 2)
        )
        closing_from_tx = None if opening_s is None else round(opening_s + in_tx - out_tx, 2)

        opening_rows = _opening_from_rows(txs)
        closing_rows = _closing_from_rows(txs)
        closing_from_tx_rows = None if opening_rows is None else round(opening_rows + in_tx - out_tx, 2)

        # NEW: consider "summary present" if opening+totals exist, regardless of closing_s
        has_summary = all(v is not None for v in (opening_s, in_s, out_s))

        print(f"\n{cur} ({n} transactions)")

        if has_summary:
            # Compare summary totals to tx sums
            d_out, ok_out = _cmp(out_s, out_tx)
            d_in,  ok_in  = _cmp(in_s, in_tx)

            # Opening check only if closing_s exists (since it uses closing_s)
            if opening_from_summary is None:
                print(f"  ⚪ Opening:       summary {_fmt(opening_s)}  → cannot derive from (closing − in + out)")
                ok_open = True  # don't penalize
                d_open = None
            else:
                d_open, ok_open = _cmp(opening_s, opening_from_summary)
                print(f"  {'✅' if ok_open else '❌'} Opening:       summary {_fmt(opening_s)}  vs  calculated-from-summary {_fmt(opening_from_summary)}  Δ {_fmt(d_open)}")

            # Money out / in vs tx
            print(f"  {'✅' if ok_out else '❌'} Money out:     summary {_fmt(out_s)}  vs  transactions-sum {_fmt(out_tx)}  Δ {_fmt(d_out)}")
            print(f"  {'✅' if ok_in  else '❌'} Money in:      summary {_fmt(in_s)}  vs  transactions-sum {_fmt(in_tx)}  Δ {_fmt(d_in)}")

            # Closing: always show calculated-from-transactions; compare to closing_s if present
            if closing_from_tx is None:
                print(f"  ⚪ Closing:       summary {_fmt(closing_s)}  → cannot derive from (opening + in(tx) − out(tx))")
                ok_close = True
                d_close = None
            else:
                if closing_s is None:
                    print(f"  ⚪ Closing:       summary None  vs  calculated-from-transactions {_fmt(closing_from_tx)}  Δ N/A")
                    ok_close = True
                    d_close = None
                else:
                    d_close, ok_close = _cmp(closing_s, closing_from_tx)
                    print(f"  {'✅' if ok_close else '❌'} Closing:       summary {_fmt(closing_s)}  vs  calculated-from-transactions {_fmt(closing_from_tx)}  Δ {_fmt(d_close)}")
                    # Optional: flag obviously broken OCR closings
                    if ok_close is False and abs(d_close) >= 1000:
                        print("  🔶 Note: closing balance looks suspiciously far from the transactions-derived closing. "
                              "Parser may have dropped digits or mis-read the final balance cell.")

            cur_ok = all([
                ok_out if ok_out is not None else True,
                ok_in  if ok_in  is not None else True,
                ok_open if opening_from_summary is not None else True,
                ok_close if d_close is not None else True,
            ])

            report["currencies"][cur] = {
                "mode": "revolut",
                "transactions": n,
                "summary": {"opening": opening_s, "in": in_s, "out": out_s, "closing": closing_s},
                "tx_sums": {"in": in_tx, "out": out_tx},
                "derived": {"opening_from_summary": opening_from_summary, "closing_from_tx": closing_from_tx},
                "deltas": {"opening": d_open, "in": d_in, "out": d_out, "closing": d_close},
                "ok": cur_ok,
            }
            report["ok"] = report["ok"] and cur_ok

        else:
            # Original BOI-style branch unchanged
            print(f"  ⚪ Opening:       opening balance {_fmt(opening_rows) if opening_rows is not None else 'N/A'}")
            print(f"  ⚪ Money out:     transactions-sum {_fmt(out_tx)}")
            print(f"  ⚪ Money in:      transactions-sum {_fmt(in_tx)}")

            if closing_rows is None and closing_from_tx_rows is None:
                print(f"  ⚪ Closing:       N/A  → cannot derive from rows")
                ok_close = None
                d_close = None
            elif closing_rows is None:
                print(f"  ⚪ Closing:       closing balance None  vs  calculated-from-transactions {_fmt(closing_from_tx_rows)}  Δ N/A")
                ok_close = None
                d_close = None
            elif closing_from_tx_rows is None:
                print(f"  ⚪ Closing:       closing balance {_fmt(closing_rows)}  → cannot derive from rows")
                ok_close = None
                d_close = None
            else:
                d_close, ok_close = _cmp(closing_rows, closing_from_tx_rows)
                print(f"  {'✅' if ok_close else '❌'} Closing:       closing balance {_fmt(closing_rows)}  vs  calculated-from-transactions {_fmt(closing_from_tx_rows)}  Δ {_fmt(d_close)}")

            cur_ok = True if ok_close in (True, None) else False

            report["currencies"][cur] = {
                "mode": "boi",
                "transactions": n,
                "summary": {"opening": opening_s, "in": in_s, "out": out_s, "closing": closing_s},
                "rows": {"opening_from_rows": opening_rows, "closing_from_rows": closing_rows},
                "tx_sums": {"in": in_tx, "out": out_tx},
                "derived": {"closing_from_tx_rows": closing_from_tx_rows},
                "deltas": {"closing": d_close},
                "ok": cur_ok,
            }
            report["ok"] = report["ok"] and cur_ok

    return report
