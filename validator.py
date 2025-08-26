# validator3.py
import math
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional

ABS_TOL = 0.01       # cents tolerance
COERCE_LEGACY = True # convert legacy currency sections into the strict structure (if needed)


# ----------------------------
# Small helpers
# ----------------------------
def _num(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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


def _tx_amount(t):
    """Return transaction amount as float, whether 'amount' is a float or {'value': ...}."""
    a = t.get("amount")
    if isinstance(a, (int, float)):
        return _num(a)
    if isinstance(a, dict):
        return _num(a.get("value"))
    return None


def _tx_sums(txs):
    """Return (in_sum, out_sum) from transactions list (amount may be float or dict)."""
    in_tx  = round(sum((_tx_amount(t) or 0.0) for t in txs if t.get("transaction_type") == "credit"), 2)
    out_tx = round(sum((_tx_amount(t) or 0.0) for t in txs if t.get("transaction_type") == "debit"),  2)
    return in_tx, out_tx


def _signed_amount(t):
    return _num(t.get("signed_amount"))


def _signed_sum(txs):
    vals = [_signed_amount(t) for t in txs if "signed_amount" in t]
    return None if not vals else round(sum(v for v in vals if v is not None), 2)


# ----------------------------
# Schema enforcement & coercion (per-currency)
# ----------------------------
STRICT_BALANCES_TEMPLATE = {
    "opening_balance":   {"summary_table": None, "transactions_table": None},
    "money_in_total":    {"summary_table": None, "transactions_table": None},
    "money_out_total":   {"summary_table": None, "transactions_table": None},
    "closing_balance":   {"summary_table": None, "transactions_table": None, "calculated": None},
}

def _ensure_balances_shape(balances: dict) -> dict:
    """Ensure the required keys exist; fill missing leaves with None."""
    shaped = deepcopy(STRICT_BALANCES_TEMPLATE)
    if not isinstance(balances, dict):
        return shaped
    for k, sub in STRICT_BALANCES_TEMPLATE.items():
        v = balances.get(k)
        if not isinstance(v, dict):
            continue
        shaped[k].update({kk: _num(v.get(kk)) for kk in sub.keys()})
    return shaped


def _coerce_legacy_currency_section(sec: dict) -> dict:
    """
    Convert a legacy currency section into the strict structure.
    Legacy examples:
      opening_balance: {value, currency} or number
      money_in_total / money_out_total: ditto
      closing_balance_statement / closing_balance_calculated
    """
    txs = sec.get("transactions", []) or []

    def _val(node):
        if node is None:
            return None
        if isinstance(node, (int, float)):
            return float(node)
        if isinstance(node, dict):
            return _num(node.get("value"))
        return None

    # derive tx sums
    in_tx, out_tx = _tx_sums(txs)

    # derive opening from rows if balance_after_statement & amount exist on the first tx
    opening_tx = None
    if txs:
        first = txs[0]
        bal = (first.get("amount_after_statement") or first.get("balance_after_statement") or {}).get("value")
        amt_node = first.get("amount")
        if isinstance(amt_node, dict):
            amt = _num(amt_node.get("value"))
        else:
            amt = _num(amt_node)
        typ = first.get("transaction_type")
        if bal is not None and amt is not None and typ in ("credit", "debit"):
            opening_tx = round(float(bal) - float(amt), 2) if typ == "credit" else round(float(bal) + float(amt), 2)

    closing_stmt_tx = None
    if txs:
        last = txs[-1]
        closing_stmt_tx = (last.get("amount_after_statement") or last.get("balance_after_statement") or {}).get("value")
        closing_stmt_tx = _num(closing_stmt_tx)

    # map legacy fields to strict leaves
    strict = deepcopy(STRICT_BALANCES_TEMPLATE)
    strict["opening_balance"]["transactions_table"] = opening_tx
    strict["money_in_total"]["transactions_table"]  = in_tx
    strict["money_out_total"]["transactions_table"] = out_tx
    strict["closing_balance"]["transactions_table"] = closing_stmt_tx

    # legacy statement/summary closings (if present)
    strict["closing_balance"]["summary_table"] = _val(sec.get("closing_balance_statement"))
    strict["closing_balance"]["calculated"]    = _val(sec.get("closing_balance_calculated"))

    # If legacy had bare numbers at top-level, treat them as transactions_table:
    ob_legacy  = _val(sec.get("opening_balance"))
    mi_legacy  = _val(sec.get("money_in_total"))
    mo_legacy  = _val(sec.get("money_out_total"))
    if strict["opening_balance"]["transactions_table"] is None and ob_legacy is not None:
        strict["opening_balance"]["transactions_table"] = ob_legacy
    if mi_legacy is not None:
        strict["money_in_total"]["transactions_table"] = mi_legacy
    if mo_legacy is not None:
        strict["money_out_total"]["transactions_table"] = mo_legacy

    return strict


def _enforce_currency_section(cur_section: dict) -> dict:
    """
    Return a new currency section that definitely has:
      - balances (strict shape)
      - transactions list
    """
    sec = deepcopy(cur_section or {})
    txs = sec.get("transactions", []) or []
    if "transactions" not in sec:
        sec["transactions"] = txs

    if "balances" in sec and isinstance(sec["balances"], dict):
        sec["balances"] = _ensure_balances_shape(sec["balances"])
    elif COERCE_LEGACY:
        sec["balances"] = _coerce_legacy_currency_section(sec)
    else:
        sec["balances"] = deepcopy(STRICT_BALANCES_TEMPLATE)

    return sec


def _enforce_statement(stmt: dict) -> dict:
    """
    Ensure every currency under stmt['currencies'] conforms to strict schema.
    Returns a deep-copied, normalized statement dict.
    """
    s = deepcopy(stmt or {})
    curr = s.get("currencies", {}) or {}
    for k in list(curr.keys()):
        curr[k] = _enforce_currency_section(curr[k])
    s["currencies"] = curr
    return s


# ----------------------------
# Per-statement validation
# ----------------------------
def validate_single_statement(statement: dict) -> dict:
    """
    Enforce schema (optionally coercing legacy), then validate ONE statement node.
    Prints a readable report and returns a structured dict.
    """
    s = _enforce_statement(statement)

    ident = s.get("file_name") or s.get("statement_id") or "<unnamed>"
    inst  = s.get("institution")
    iban  = s.get("iban")
    acct  = s.get("account_type")
    print(f"\n=== VALIDATION (balances) for: {ident} ===")
    if inst or iban or acct:
        extra = " | ".join(x for x in [inst, iban, acct] if x)
        if extra:
            print(extra)

    report = {
        "file_name": s.get("file_name"),
        "statement_id": s.get("statement_id"),
        "institution": inst,
        "iban": iban,
        "account_type": acct,
        "statement_start_date": s.get("statement_start_date"),
        "statement_end_date": s.get("statement_end_date"),
        "currencies": {},
        "ok": True,
    }

    currencies = s.get("currencies", {}) or {}
    for cur in sorted(currencies.keys()):
        sec = currencies[cur] or {}
        txs = sec.get("transactions", []) or []
        n = len(txs)

        b = sec.get("balances", {}) or {}
        ob = b.get("opening_balance", {}) or {}
        mi = b.get("money_in_total", {}) or {}
        mo = b.get("money_out_total", {}) or {}
        cb = b.get("closing_balance", {}) or {}

        open_sum = _num(ob.get("summary_table"))
        open_tx  = _num(ob.get("transactions_table"))

        in_sum   = _num(mi.get("summary_table"))
        in_tx_t  = _num(mi.get("transactions_table"))

        out_sum  = _num(mo.get("summary_table"))
        out_tx_t = _num(mo.get("transactions_table"))

        close_sum  = _num(cb.get("summary_table"))
        close_tx   = _num(cb.get("transactions_table"))
        close_calc = _num(cb.get("calculated"))

        # True tx sums (from amounts)
        in_tx, out_tx = _tx_sums(txs)


        pu = ((s.get("meta") or {}).get("page_usage") or {})
        if pu:
            nt = pu.get("native_text_pages", 0)
            oc = pu.get("ocr_pages", 0)
            if nt != 0 and oc != 0:
                print(f"\n{cur} ({n} transactions from {nt} text & {oc} ocr pages)")
            elif nt != 0:
                print(f"\n{cur} ({n} transactions from {nt} text pages)")
            else:
                print(f"\n{cur} ({n} transactions from {oc} ocr pages)")
        else:
            print(f"\n{cur} ({n} transactions)")

        # Opening
        if open_sum is None and open_tx is None:
            print("  ✅ Opening: no summary table and no transactions-table opening (acceptable).")
        else:
            if open_sum is not None and open_tx is not None:
                d_open, ok_open = _cmp(open_sum, open_tx)
                print(f"  {'✅' if ok_open else 'ℹ️'} Opening: summary_table {_fmt(open_sum)} vs transactions_table {_fmt(open_tx)}  Δ {_fmt(d_open)}")
            elif open_sum is not None:
                print(f"  ✅ Opening: summary_table {_fmt(open_sum)}")
            else:
                print(f"  ✅ Opening: transactions_table {_fmt(open_tx)}")

        # Money out / in — compare tx totals vs transactions_table (ground truth if present)
        d_out_tx, ok_out_tx = _cmp(out_tx, out_tx_t if out_tx_t is not None else out_tx)
        d_in_tx,  ok_in_tx  = _cmp(in_tx,  in_tx_t  if in_tx_t  is not None else in_tx)

        print(f"  {'✅' if ok_out_tx else '❌'} Money out: transactions_table {_fmt(out_tx_t)} vs tx-sum {_fmt(out_tx)}  Δ {_fmt(d_out_tx)}")
        if out_sum is None:
            print(f"  ✅ Money out: no summary table")
        else:
            d_out_sum, _ = _cmp(out_sum, out_tx)
            print(f"  ℹ️  Money out: summary_table {_fmt(out_sum)} vs tx-sum {_fmt(out_tx)}  Δ {_fmt(d_out_sum)}")

        print(f"  {'✅' if ok_in_tx else '❌'} Money in:  transactions_table {_fmt(in_tx_t)} vs tx-sum {_fmt(in_tx)}   Δ {_fmt(d_in_tx)}")
        if in_sum is None:
            print(f"  ✅ Money in:  no summary table")
        else:
            d_in_sum, _ = _cmp(in_sum, in_tx)
            print(f"  ℹ️  Money in:  summary_table {_fmt(in_sum)} vs tx-sum {_fmt(in_tx)}   Δ {_fmt(d_in_sum)}")

        # Optional signed_amount checks
        signed_total = _signed_sum(txs)
        signed_ok = None
        if signed_total is not None:
            # sign consistency and absolute-magnitude consistency
            sign_mismatch = 0
            abs_mismatch = 0
            for t in txs:
                if "signed_amount" not in t:
                    continue
                sa = _signed_amount(t)
                amt = _tx_amount(t)
                typ = t.get("transaction_type")
                if sa is None or amt is None or typ not in ("credit", "debit"):
                    continue
                expected_sign = 1 if typ == "credit" else -1
                if expected_sign * 1.0 * amt is None:
                    continue
                # sign check
                if (sa > 0 and typ == "debit") or (sa < 0 and typ == "credit"):
                    sign_mismatch += 1
                # magnitude check
                if not math.isclose(abs(sa), float(amt), abs_tol=ABS_TOL):
                    abs_mismatch += 1

            net_expected = round(in_tx - out_tx, 2)
            d_signed, ok_signed_net = _cmp(net_expected, signed_total)
            signed_ok = (sign_mismatch == 0 and abs_mismatch == 0 and (ok_signed_net is True))

            print(f"  {'✅' if signed_ok else '❌'} Signed amounts: "
                  f"net {_fmt(signed_total)} vs (in - out) {_fmt(net_expected)}  Δ {_fmt(d_signed)}; "
                  f"sign mismatches={sign_mismatch}, abs mismatches={abs_mismatch}")

        # Closing
        calc_from_tx = None
        if open_tx is not None:
            calc_from_tx = round(open_tx + in_tx - out_tx, 2)
        elif open_sum is not None:
            calc_from_tx = round(open_sum + in_tx - out_tx, 2)

        any_close_info = False
        if close_calc is not None and calc_from_tx is not None:
            d_close_calc, ok_close_calc = _cmp(close_calc, calc_from_tx)
            print(f"  {'✅' if ok_close_calc else '❌'} Closing (calculated): model {_fmt(close_calc)} vs derived-from-tx {_fmt(calc_from_tx)}  Δ {_fmt(d_close_calc)}")
            any_close_info = True

        if close_sum is not None and calc_from_tx is not None:
            d_close_sum, ok_close_sum = _cmp(close_sum, calc_from_tx)
            print(f"  {'✅' if ok_close_sum else '❌'} Closing (summary_table): {_fmt(close_sum)} vs derived-from-tx {_fmt(calc_from_tx)}  Δ {_fmt(d_close_sum)}")
            any_close_info = True

        if close_tx is not None and calc_from_tx is not None:
            d_close_tx, ok_close_tx = _cmp(close_tx, calc_from_tx)
            print(f"  {'✅' if ok_close_tx else '❌'} Closing (transactions_table): {_fmt(close_tx)} vs derived-from-tx {_fmt(calc_from_tx)}  Δ {_fmt(d_close_tx)}")
            any_close_info = True

        if not any_close_info:
            print("  ⚪ Closing: missing values")

        # overall ok (require tx totals to match; include signed check if present)
        cur_ok = not ((ok_out_tx is False) or (ok_in_tx is False))
        if signed_ok is False:
            cur_ok = False

        report["ok"] = report["ok"] and cur_ok
        report["currencies"][cur] = {
            "transactions": n,
            "balances": sec.get("balances"),
            "tx_sums": {"in": in_tx, "out": out_tx, "signed_net": signed_total},
            "ok": cur_ok,
        }

    return report


# ----------------------------
# Bundle-level validation
# ----------------------------
def validate_single_client_bundle(bundle: dict) -> dict:
    """
    Validate a legacy single-client bundle:
      { "schema_version": "...", "client": "...", "statements": [ <statement> ... ] }
    """
    stmts = bundle.get("statements", []) or []
    client = bundle.get("client")
    if client:
        print(f"\n### CLIENT: {client}")

    overall_ok = True
    per_statement_reports: List[dict] = []

    for stmt in stmts:
        rep = validate_single_statement(stmt)
        per_statement_reports.append(rep)
        overall_ok = overall_ok and bool(rep.get("ok", False))

    return {
        "client": client,
        "schema_version": bundle.get("schema_version"),
        "ok": overall_ok,
        "statements": per_statement_reports,
    }


def validate_clients_bundle(bundle: dict) -> dict:
    """
    Validate the new multi-client bundle:
      {
        "schema_version": "...",
        "clients": [
          {"name": "...", "statements": [ <statement> ... ]},
          ...
        ]
      }
    """
    clients = bundle.get("clients", []) or []
    print("\n### VALIDATING MULTI-CLIENT BUNDLE")
    overall_ok = True
    client_reports: List[dict] = []

    for c in clients:
        name = c.get("name") or "<unnamed>"
        print(f"\n#### CLIENT: {name}")
        stmts = c.get("statements", []) or []

        per_statement_reports: List[dict] = []
        client_ok = True
        for stmt in stmts:
            rep = validate_single_statement(stmt)
            per_statement_reports.append(rep)
            client_ok = client_ok and bool(rep.get("ok", False))

        overall_ok = overall_ok and client_ok
        client_reports.append({
            "name": name,
            "ok": client_ok,
            "statements": per_statement_reports,
        })

    return {
        "schema_version": bundle.get("schema_version"),
        "ok": overall_ok,
        "clients": client_reports,
    }


# ----------------------------
# Auto-detect entrypoint
# ----------------------------
def validate(data: dict) -> dict:
    """
    Convenience entrypoint:
      - If 'clients' is present -> multi-client bundle
      - Else if 'statements' is present -> legacy single-client bundle
      - Else -> single-statement mode
    """
    if isinstance(data, dict) and "clients" in data:
        return validate_clients_bundle(data)
    if isinstance(data, dict) and "statements" in data:
        print(f"\n### VALIDATING SINGLE-CLIENT BUNDLE")
        return validate_single_client_bundle(data)
    print(f"\n### VALIDATING SINGLE STATEMENT")
    return validate_single_statement(data)
