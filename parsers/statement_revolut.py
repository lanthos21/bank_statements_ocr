import re
import math
import pandas as pd
from datetime import datetime
from utils import parse_currency, parse_date


def detect_column_positions_from_lines(lines: list[dict]) -> tuple[dict, int, int]:
    for idx, line in enumerate(lines):
        line_text = " ".join(w["text"] for w in line["words"] if "text" in w).lower()
        if (
            "date" in line_text
            and "description" in line_text
            and "money out" in line_text
            and "money in" in line_text
        ):
            # print(f"🧭 Transaction header found on line {idx}: {line_text}")

            positions = {
                "date": None,
                "description": None,
                "money_out": None,
                "money_in": None,
                "balance": None,
            }

            # print("🧩 Words on header line:")
            for i, word in enumerate(line["words"]):
                text = word.get("text", "").lower()
                x = word.get("x", word.get("left"))

                # print(f"  - {word.get('text')} @ x={x}")

                if "date" in text:
                    positions["date"] = x
                elif "description" in text:
                    positions["description"] = x
                elif text == "balance":
                    positions["balance"] = x

                # Detect 'money out' and 'money in' pairs
                if text == "money" and i + 1 < len(line["words"]):
                    next_word = line["words"][i + 1]
                    next_text = next_word.get("text", "").lower()
                    next_x = next_word.get("x", next_word.get("left"))

                    if next_text == "out":
                        positions["money_out"] = x  # or next_x, depending on your alignment
                    elif next_text == "in":
                        positions["money_in"] = x  # or next_x

            # print("📌 Header positions:", positions)
            return {
                "out": positions["money_out"],
                "in": positions["money_in"],
                "bal": positions["balance"]
            }, idx, len(lines)

    raise ValueError("❌ Could not find transaction header row.")


def categorise_revolut_amount_by_left_edge(x1, header_positions, margin=100):
    for category in ["out", "in", "bal"]:
        pos = header_positions.get(category)
        if pos is not None and abs(x1 - pos) <= margin:
            return category
    return "unknown"


def extract_revolut_date_range(pages: list[dict]) -> tuple[str | None, str | None]:
    pattern = r"from (\d{1,2} \w+ \d{4}) to (\d{1,2} \w+ \d{4})"
    for page in pages:
        for line in page["lines"]:
            match = re.search(pattern, line["line_text"], re.IGNORECASE)
            if match:
                return match.group(1), match.group(2)
    return None, None


def parse_revolut_transactions_from_coords(pages: list[dict], iban: str | None = None) -> list[dict]:
    all_transactions = []
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    running_balance_cents = None
    start_date_str, end_date_str = extract_revolut_date_range(pages)
    initial_value_date = parse_date(start_date_str) if start_date_str else None

    # ✅ New accumulators
    total_money_in = 0.0
    total_money_out = 0.0

    for page_num, page in enumerate(pages):
        # print(f"\n📄 Processing Page {page_num + 1}")
        lines = page["lines"]

        try:
            header_positions, start_idx, end_idx = detect_column_positions_from_lines(lines)
            # print("📌 Header positions:", header_positions)
        except ValueError as e:
            # print(str(e))
            continue

        # print("\n🔍 Scanning for '15 Jan 2025' in line_texts...")
        lines_to_check = lines[start_idx + 1: end_idx]

        for i, line in enumerate(lines_to_check):
            if "5 Mar 2025xxxxx" in line["line_text"]:
                print(f"\n🧪 FOUND MATCH at index {i} in page lines")
                print("📄 Current line:", line["line_text"])
                print("📊 Words:", [w["text"] for w in line["words"]])

                # Print previous line
                if i > 0:
                    prev = lines_to_check[i - 3]
                    print("\n⬆️ Previous line:")
                    print("📄", prev["line_text"])
                    print("📊", [w["text"] for w in prev["words"]])
                if i > 0:
                    prev = lines_to_check[i - 2]
                    print("\n⬆️ Previous line:")
                    print("📄", prev["line_text"])
                    print("📊", [w["text"] for w in prev["words"]])
                if i > 0:
                    prev = lines_to_check[i - 1]
                    print("\n⬆️ Previous line:")
                    print("📄", prev["line_text"])
                    print("📊", [w["text"] for w in prev["words"]])


        last_seen_date = initial_value_date

        for line in lines[start_idx + 1 : end_idx]:
            df = pd.DataFrame(line["words"])

            if df.empty:
                # print("⚠️ Empty dataframe for line:", line["line_text"])
                continue

            df.rename(columns={"word": "text", "x": "left"}, inplace=True)
            df["line_text"] = line["line_text"]
            df["amount"] = df["text"].apply(parse_currency)
            df["category"] = df["left"].apply(lambda x: categorise_revolut_amount_by_left_edge(x, header_positions))

            # 🔍 Require a valid date on the line (no fallback to last_seen_date)
            date_match = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", line["line_text"])
            if not date_match:
                # print(f"⚠️ Skipping line (no date found): {line['line_text']}")
                continue

            parsed_date = parse_date(date_match.group(0))
            if not parsed_date:
                # print(f"⚠️ Skipping line (date failed to parse): {line['line_text']}")
                continue

            transaction_date = parsed_date

            # Get amounts
            in_series = df[df["category"] == "in"]["amount"]
            out_series = df[df["category"] == "out"]["amount"]
            bal_series = df[df["category"] == "bal"]["amount"]

            credit_amount = in_series.iloc[0] if not in_series.empty else None
            debit_amount = out_series.iloc[0] if not out_series.empty else None
            balance = bal_series.iloc[0] if not bal_series.empty else None

            # Must have date, balance, and either in or out
            if not transaction_date or (credit_amount is None and debit_amount is None) or balance is None:
                # print(f"⚠️ Skipping line due to missing required fields: {line['line_text']}")
                continue

            # Clean description: remove date + amounts
            known_amounts = set(df[~df["amount"].isna()]["text"])
            date_parts = date_match.group(0).split() if date_match else []
            clean_desc = " ".join([
                word for word in line["line_text"].split()
                if word not in known_amounts and word not in date_parts
            ])

            # Normalize values
            credit_amount = 0.0 if credit_amount is None or math.isnan(credit_amount) else credit_amount
            debit_amount = 0.0 if debit_amount is None or math.isnan(debit_amount) else debit_amount

            # ✅ Accumulate totals
            total_money_in += credit_amount
            total_money_out += debit_amount

            transaction_type = "credit" if credit_amount > 0 else "debit"
            credit_amount_cents = int(round(credit_amount * 100))
            debit_amount_cents = int(round(debit_amount * 100))

            # Running balance with discrepancy handling
            if running_balance_cents is None:
                running_balance_cents = int(round(balance * 100))
            else:
                running_balance_cents += credit_amount_cents - debit_amount_cents

            calculated_balance = round(running_balance_cents / 100, 2)

            # Check for discrepancy
            if not math.isclose(calculated_balance, balance, abs_tol=0.01):
                discrepancy = round(balance - calculated_balance, 2)
                print("\n⚠️ BALANCE DISCREPANCY DETECTED")
                print(f"📆 Date: {transaction_date}")
                print(f"📝 Description: {clean_desc.strip()}")
                print(f"💸 Credit: {credit_amount:.2f} | Debit: {debit_amount:.2f}")
                print(f"📊 Calculated Balance: {calculated_balance:.2f}")
                print(f"📄 Statement Balance:  {balance:.2f}")
                print(f"🧮 Discrepancy: {discrepancy:+.2f}")
                print(f"📄 Full OCR Line: {line['line_text']}")

                # Fix the calculated balance
                #running_balance_cents = int(round(balance * 100))
                #calculated_balance = round(running_balance_cents / 100, 2)

            transaction = {
                "transactions_date": transaction_date,
                "transaction_type": transaction_type,
                "description": clean_desc.strip(),
                "amount": {"value": credit_amount if credit_amount > 0 else debit_amount, "currency": currency},
                "balance_after_statement": {"value": balance, "currency": currency},
                "balance_after_calculated": {"value": calculated_balance, "currency": currency},
            }
            all_transactions.append(transaction)

    # ✅ Print totals at the end
    print(f"\n📄 TOTAL REVOLUT TRANSACTIONS FOUND: {len(all_transactions)}")
    print(f"💰 TOTAL MONEY IN:  {total_money_in:.2f} {currency}")
    print(f"💸 TOTAL MONEY OUT: {total_money_out:.2f} {currency}")

    return all_transactions


def parse_revolut_statement(raw_ocr, client="Unknown", account_type="Unknown"):
    full_text = "\n".join("\n".join(line["line_text"] for line in page["lines"]) for page in raw_ocr["pages"])

    iban_match = re.search(r"IBAN\s+([A-Z]{2}\d{2}[A-Z0-9]{11,30})", full_text)
    iban = iban_match.group(1).strip() if iban_match else None
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    start_date_str, end_date_str = extract_revolut_date_range(raw_ocr["pages"])
    start_date = parse_date(start_date_str) if start_date_str else None
    end_date = parse_date(end_date_str) if end_date_str else None

    transactions = parse_revolut_transactions_from_coords(raw_ocr["pages"], iban=iban)

    return {
        "client": client,
        "file_name": raw_ocr["file_name"],
        "account_holder": None,
        "institution": "Revolut",
        "account_type": account_type,
        "iban": iban,
        "bic": None,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "opening_balance": None,
        "closing_balance_statement": transactions[-1]["balance_after_statement"] if transactions else None,
        "closing_balance_calculated": transactions[-1]["balance_after_calculated"] if transactions else None,
        "transactions": transactions,
    }
