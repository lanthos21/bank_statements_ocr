import math

ABS_TOL = 0.01  # cents tolerance

def _num(x):
    return None if x is None else float(x)

def _val(node):
    return None if not node else _num(node.get("value"))

def validate_statement_json(data: dict) -> dict:
    """
    Validate a parsed Revolut statement JSON by currency, using only the final JSON.
    Checks:
      - Opening (summary) ≈ Closing(summary) - In(summary) + Out(summary)
      - Money In (summary) ≈ Sum of tx credits
      - Money Out (summary) ≈ Sum of tx debits
      - Closing (summary) ≈ Opening(summary) + In(tx) - Out(tx)
    Prints a compact report and returns a structured dict.
    """
    report = {"file_name": data.get("file_name"), "currencies": {}, "ok": True}

    print(f"\n=== VALIDATION for: {data.get('file_name')} ===")
    currencies = data.get("currencies", {}) or {}

    for cur in sorted(currencies.keys()):
        sec = currencies[cur]
        txs = sec.get("transactions", []) or []
        n = len(txs)

        # summary (declared) numbers
        opening_s  = _val(sec.get("opening_balance"))
        in_s       = _val(sec.get("money_in_total"))
        out_s      = _val(sec.get("money_out_total"))
        closing_s  = _val(sec.get("closing_balance_statement"))

        # transaction sums
        in_tx  = sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "credit")
        out_tx = sum(float(t["amount"]["value"]) for t in txs if t.get("transaction_type") == "debit")

        # derived values for checks
        # Opening check compares summary opening vs (summary closing - in + out)
        opening_from_summary = None
        if closing_s is not None and in_s is not None and out_s is not None:
            opening_from_summary = round(closing_s - in_s + out_s, 2)

        # Closing check compares summary closing vs (summary opening + in_tx - out_tx)
        closing_from_tx = None
        if opening_s is not None:
            closing_from_tx = round(opening_s + in_tx - out_tx, 2)

        # Tolerance checks
        def _cmp(a, b):
            if a is None or b is None:
                return None, False
            return round(b - a, 2), math.isclose(a, b, abs_tol=ABS_TOL)

        # Opening line
        open_delta, open_ok = _cmp(opening_s, opening_from_summary)

        # Money in/out lines
        in_delta, in_ok   = _cmp(in_s, in_tx)
        out_delta, out_ok = _cmp(out_s, out_tx)

        # Closing line
        close_delta, close_ok = _cmp(closing_s, closing_from_tx)

        # Pretty print
        def flag(ok):
            if ok is None: return "⚪"
            return "✅" if ok else "❌"

        print(f"\n{cur} ({n} transactions)")
        # Opening
        if opening_from_summary is None:
            print(f"  {flag(open_ok)} Opening:       summary {opening_s}  → cannot derive from (closing − in + out)")
        else:
            print(f"  {flag(open_ok)} Opening:       summary {opening_s}  vs  calculated-from-summary {opening_from_summary}  Δ {open_delta:+.2f}")
        # Money out
        if out_s is None:
            print(f"  {flag(out_ok)} Money out:     summary None  vs  tx-sum {out_tx:.2f}  ⚪")
        else:
            print(f"  {flag(out_ok)} Money out:     summary {out_s:.2f}  vs  transactions-sum {out_tx:.2f}  Δ {out_delta:+.2f}")
        # Money in
        if in_s is None:
            print(f"  {flag(in_ok)} Money in:      summary None  vs  tx-sum {in_tx:.2f}  ⚪")
        else:
            print(f"  {flag(in_ok)} Money in:      summary {in_s:.2f}   vs  transactions-sum {in_tx:.2f}   Δ {in_delta:+.2f}")
        # Closing
        if closing_from_tx is None:
            print(f"  {flag(close_ok)} Closing:       summary {closing_s}  → cannot derive from (opening + in(tx) − out(tx))")
        else:
            print(f"  {flag(close_ok)} Closing:       summary {closing_s}  vs  calculated-from-transactions {closing_from_tx}  Δ {close_delta:+.2f}")

        cur_ok = all([
            (open_ok if opening_from_summary is not None else True),
            (in_ok   if in_s is not None else True),
            (out_ok  if out_s is not None else True),
            (close_ok if closing_from_tx is not None else True),
        ])
        report["currencies"][cur] = {
            "transactions": n,
            "summary": {
                "opening": opening_s,
                "in": in_s,
                "out": out_s,
                "closing": closing_s,
            },
            "tx_sums": {
                "in": round(in_tx, 2),
                "out": round(out_tx, 2),
            },
            "derived": {
                "opening_from_summary": opening_from_summary,
                "closing_from_tx": closing_from_tx,
            },
            "deltas": {
                "opening": open_delta,
                "in": in_delta,
                "out": out_delta,
                "closing": close_delta,
            },
            "ok": cur_ok,
        }
        report["ok"] = report["ok"] and cur_ok

    #print(f"\nOVERALL: {'✅ All good' if report['ok'] else '❌ Issues found'}")
    return report
