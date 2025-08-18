# validator2.py
import math

ABS_TOL = 0.01  # cents tolerance


def _num(x):
    return None if x is None else float(x)


def _fmt(x):
    return "None" if x is None else f"{float(x):.2f}"


def _cmp(a, b):
    """Compare with tolerance; return (delta, ok)"""
    if a is None or b is None:
        return None, None
    d = round(float(b) - float(a), 2)
    return d, math.isclose(float(a), float(b), abs_tol=ABS_TOL)


def validate_statement_json(data: dict) -> dict:
    print(f"\n=== VALIDATION (balances) for: {data.get('file_name')} ===")

    report = {"file_name": data.get("file_name"), "currencies": {}, "ok": True}
    currencies = data.get("currencies", {}) or {}

    for cur in sorted(currencies.keys()):
        sec = currencies[cur]
        balances = sec.get("balances", {}) or {}
        txs = sec.get("transactions", []) or []
        n = len(txs)

        print(f"\n{cur} ({n} transactions)")

        # --- Opening ---
        opening_s = _num(balances.get("opening_balance", {}).get("summary_table"))
        opening_t = _num(balances.get("opening_balance", {}).get("transactions_table"))
        d_open, ok_open = _cmp(opening_s, opening_t)
        if opening_s is None and opening_t is not None:
            print(f"  ✅ Opening: no summary table → trusted from transactions_table {_fmt(opening_t)}")
        elif opening_s is not None and opening_t is not None:
            print(f"  {'✅' if ok_open else '❌'} Opening: summary {_fmt(opening_s)} vs transactions_table {_fmt(opening_t)} Δ {_fmt(d_open)}")
        else:
            print(f"  ⚪ Opening: missing values")
            ok_open = True

        # --- Money out ---
        out_s = _num(balances.get("money_out_total", {}).get("summary_table"))
        out_t = _num(balances.get("money_out_total", {}).get("transactions_table"))
        d_out, ok_out = _cmp(out_s, out_t)
        if out_s is None and out_t is not None:
            print(f"  ✅ Money out: no summary table → transactions_table {_fmt(out_t)}")
        elif out_s is not None and out_t is not None:
            print(f"  {'✅' if ok_out else '❌'} Money out: summary {_fmt(out_s)} vs transactions_table {_fmt(out_t)} Δ {_fmt(d_out)}")
        else:
            print(f"  ⚪ Money out: missing values")
            ok_out = True

        # --- Money in ---
        in_s = _num(balances.get("money_in_total", {}).get("summary_table"))
        in_t = _num(balances.get("money_in_total", {}).get("transactions_table"))
        d_in, ok_in = _cmp(in_s, in_t)
        if in_s is None and in_t is not None:
            print(f"  ✅ Money in: no summary table → transactions_table {_fmt(in_t)}")
        elif in_s is not None and in_t is not None:
            print(f"  {'✅' if ok_in else '❌'} Money in: summary {_fmt(in_s)} vs transactions_table {_fmt(in_t)} Δ {_fmt(d_in)}")
        else:
            print(f"  ⚪ Money in: missing values")
            ok_in = True

        # --- Closing ---
        closing_s = _num(balances.get("closing_balance", {}).get("summary_table"))
        closing_t = _num(balances.get("closing_balance", {}).get("transactions_table"))
        closing_c = _num(balances.get("closing_balance", {}).get("calculated"))

        if closing_s is None and closing_t is not None and closing_c is not None:
            d_close, ok_close = _cmp(closing_t, closing_c)
            print(f"  {'✅' if ok_close else '❌'} Closing: transactions_table {_fmt(closing_t)} vs calculated {_fmt(closing_c)} Δ {_fmt(d_close)}")
        elif closing_s is not None and closing_t is not None:
            d_close, ok_close = _cmp(closing_s, closing_t)
            print(f"  {'✅' if ok_close else '❌'} Closing: summary {_fmt(closing_s)} vs transactions_table {_fmt(closing_t)} Δ {_fmt(d_close)}")
            if closing_c is not None:
                d_calc, ok_calc = _cmp(closing_t, closing_c)
                print(f"  {'✅' if ok_calc else '❌'}   cross-check: transactions_table {_fmt(closing_t)} vs calculated {_fmt(closing_c)} Δ {_fmt(d_calc)}")
                ok_close = ok_close and ok_calc
        elif closing_t is not None:
            print(f"  ✅ Closing: no summary table → transactions_table {_fmt(closing_t)} (calculated {_fmt(closing_c)})")
            ok_close = True
        else:
            print(f"  ⚪ Closing: missing values")
            ok_close = True

        cur_ok = all([ok_open, ok_out, ok_in, ok_close])
        report["currencies"][cur] = {
            "transactions": n,
            "ok": cur_ok,
        }
        report["ok"] = report["ok"] and cur_ok

    return report
