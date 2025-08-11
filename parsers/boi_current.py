import re
import pandas as pd
from datetime import datetime
from utils import parse_currency, parse_date


def detect_column_positions(lines: list[dict]) -> tuple[dict, int, int]:
    """Locate column headers like 'Payments - in', 'Payments - out', 'Balance'"""
    header_positions = {}
    header_start_idx = None

    def find_phrase_x_right(line, phrase: str) -> int | None:
        words = line["words"]
        phrase_parts = phrase.lower().split()
        for i in range(len(words) - len(phrase_parts) + 1):
            match = all(
                phrase_parts[j] in words[i + j]["text"].lower()
                for j in range(len(phrase_parts))
            )
            if match:
                final_word = words[i + len(phrase_parts) - 1]
                return final_word["left"] + final_word["width"]
        return None

    for idx, line in enumerate(lines):
        if not line["words"]:
            continue

        # Try to locate all headers
        right_out = find_phrase_x_right(line, "Payments - out")
        right_in = find_phrase_x_right(line, "Payments - in")
        right_bal = next(
            (w["left"] + w["width"] for w in line["words"] if "balance" in w["text"].lower()),
            None
        )

        # If all headers found on this line
        if right_out and right_in and right_bal:
            header_positions["out"] = right_out
            header_positions["in"] = right_in
            header_positions["bal"] = right_bal
            header_start_idx = idx
            print(f"🧭 'Payments - out' at x={right_out} on line {idx}")
            print(f"🧭 'Payments - in' at x={right_in} on line {idx}")
            print(f"🧭 'Balance' at x={right_bal} on line {idx}")
            break

    if header_start_idx is None:
        print("⚠️ Header row not found on this page")
        return None

    return header_positions, header_start_idx, len(lines)


def categorise_amount_by_right_edge(x1, header_positions, margin=100):
    for category in ["out", "in", "bal"]:
        if category in header_positions and abs(x1 - header_positions[category]) <= margin:
            return category
    return "unknown"


def extract_opening_balance_and_start_date(pages: list[dict]) -> tuple[float | None, str | None]:
    for page in pages:
        header_result = detect_column_positions(page["lines"])
        if not header_result:
            continue

        header_positions, _, _ = header_result

        for line in page["lines"]:
            if "balance forward" in line["line_text"].lower():
                df = pd.DataFrame(line["words"])
                if df.empty:
                    continue

                df["right"] = df["left"] + df["width"]
                #df["amount"] = df["text"].apply(parse_currency, False)
                df["amount"] = df["text"].apply(lambda v: parse_currency(v, strip_currency=False))
                df["category"] = df["right"].apply(lambda x: categorise_amount_by_right_edge(x, header_positions))

                bal_series = df[df["category"] == "bal"]["amount"]
                balance = round(bal_series.iloc[0], 2) if not bal_series.empty else None

                # Try to extract the start date from the line text
                date_match = re.search(r"\d{1,2} \w{3,9} \d{4}", line["line_text"])
                start_date = None
                if date_match:
                    try:
                        dt = datetime.strptime(date_match.group(0), "%d %b %Y")
                    except ValueError:
                        try:
                            dt = datetime.strptime(date_match.group(0), "%d %B %Y")
                        except ValueError:
                            dt = None

                    if dt:
                        start_date = dt.date().isoformat()

                return balance, start_date

    return None, None


def extract_iban(pages: list[dict]) -> str | None:
    iban_pattern = re.compile(r'\b([A-Z]{2}\d{2}[A-Z0-9]{11,30})\b')

    for page in pages:
        for line in page["lines"]:
            line_text = line["line_text"]
            if "IBAN" in line_text.upper():
                print(f"🔍 Found line with IBAN: {line_text}")  # DEBUG

                # Get everything after "IBAN"
                after_iban = line_text.upper().split("IBAN", 1)[-1]

                # Remove common whitespace and re-join
                cleaned = re.sub(r'\s+', '', after_iban)

                # Extract valid IBAN (strict length check)
                match = iban_pattern.search(cleaned)
                if match:
                    iban = match.group(1)
                    print(f"✅ Extracted IBAN: {iban}")  # DEBUG
                    return iban

    print("❌ No IBAN found.")
    return None


def parse_transactions(
    pages: list[dict],
    opening_balance_cents: int | None = None,
    iban: str | None = None
) -> list[dict]:

    all_transactions = []
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"

    # Look for the first available date anywhere in the document
    initial_value_date = None
    for page in pages:
        for line in page["lines"]:
            date_match = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", line["line_text"])
            if date_match:
                try:
                    date_str = date_match.group(0)
                    initial_value_date = datetime.strptime(date_str, "%d %B %Y").date().isoformat()
                    break
                except ValueError:
                    continue
        if initial_value_date:
            break

    running_balance_cents = opening_balance_cents

    for page_num, page in enumerate(pages):
        print(f"\n📄 Processing Page {page_num + 1}")
        lines = page["lines"]
        result = detect_column_positions(lines)
        if result is None:
            print(f"⚠️ Skipping Page {page_num + 1} — no header row found")
            continue

        header_positions, start_idx, end_idx = result
        print(f"🧭 Detected header positions: {header_positions}")

        last_seen_date = initial_value_date
        for line in lines[start_idx + 1:end_idx]:
            df = pd.DataFrame(line["words"])
            if df.empty:
                continue

            df["right"] = df["left"] + df["width"]
            df["line_text"] = line["line_text"]
            amount_like = df[df["text"].str.match(r"^-?\d{1,3}(,\d{3})*(\.\d{2})?$|^-?\d+\.\d{2}$")]
            if amount_like.empty:
                continue

            #df["amount"] = df["text"].apply(parse_currency, False)
            df["amount"] = df["text"].apply(lambda v: parse_currency(v, strip_currency=False))

            df["category"] = df["right"].apply(lambda x: categorise_amount_by_right_edge(x, header_positions))

            # Update last seen date
            date_match = re.search(r"\b(\d{1,2}) (\w{3,9}) (\d{4})\b", line["line_text"])
            if date_match:
                try:
                    date_str = date_match.group(0)
                    last_seen_date = datetime.strptime(date_str, "%d %B %Y").date().isoformat()
                except ValueError:
                    pass

            value_date = last_seen_date

            in_series = df[df["category"] == "in"]["amount"]
            out_series = df[df["category"] == "out"]["amount"]
            bal_series = df[df["category"] == "bal"]["amount"]

            credit_amount = in_series.iloc[0] if not in_series.empty and pd.notna(in_series.iloc[0]) else 0.0
            debit_amount = out_series.iloc[0] if not out_series.empty and pd.notna(out_series.iloc[0]) else 0.0
            if credit_amount == 0.0 and debit_amount == 0.0:
                continue  # skip balance-only or malformed rows

            credit_amount_cents = int(round(credit_amount * 100))
            debit_amount_cents = int(round(debit_amount * 100))

            transaction_type = "credit" if not in_series.empty else "debit"

            # Statement balance if shown
            bal = bal_series.iloc[0] if not bal_series.empty else None

            # Running calculated balance
            if running_balance_cents is not None:
                running_balance_cents += credit_amount_cents - debit_amount_cents
                calculated_balance = round(running_balance_cents / 100, 2)
            else:
                calculated_balance = None

            # Clean up description
            clean_desc = line["line_text"]
            if date_match:
                clean_desc = clean_desc.replace(date_match.group(0), "").strip()

            transaction = {
                "transactions_date": value_date,
                "transaction_type": transaction_type,
                "description": clean_desc,
                "amount": {"value": credit_amount if credit_amount else debit_amount, "currency": currency},
                "balance_after_statement": {"value": bal, "currency": currency} if bal is not None else None,
                "balance_after_calculated": {"value": calculated_balance, "currency": currency} if calculated_balance is not None else None,
            }
            all_transactions.append(transaction)

    print(f"\n📄 TOTAL TRANSACTIONS FOUND: {len(all_transactions)}")
    return all_transactions


def parse_statement(raw_ocr, client="Unknown", account_type="Unknown"):
    """Main entry point: Parses BOI statement from OCR."""
    full_text = "\n".join("\n".join(line["line_text"] for line in page["lines"]) for page in raw_ocr["pages"])


    # Extract metadata
    iban = extract_iban(raw_ocr["pages"])
    currency = "GBP" if iban and iban.upper().startswith("GB") else "EUR"
    bic_match = re.search(r"Bank\s+Identifier\s+Code\s+([A-Z]{6}[A-Z0-9]{2,5})", full_text, re.IGNORECASE)
    bic_value = bic_match.group(1).strip() if bic_match else None
    account_holder_match = re.search(r"(?i)Your account name\s+([^\n\r]+)", full_text)
    statement_date_match = re.search(r"Statement date\s+(\d{1,2} \w{3} \d{4})", full_text)

    opening_balance, start_date = extract_opening_balance_and_start_date(raw_ocr["pages"])
    end_date = parse_date(statement_date_match.group(1)) if statement_date_match else None

    opening_balance_cents = int(round(opening_balance * 100)) if opening_balance else None
    transactions = parse_transactions(raw_ocr["pages"], opening_balance_cents=opening_balance_cents, iban=iban)


    return {
        "client": client,
        "file_name": raw_ocr["file_name"],
        "account_holder": account_holder_match.group(1).strip() if account_holder_match else None,
        "institution": "Bank of Ireland",
        "account_type": account_type,
        "iban": iban,
        "bic": bic_value,
        "statement_start_date": start_date,
        "statement_end_date": end_date,
        "opening_balance": {"value": opening_balance, "currency": currency} if opening_balance else None,
        "closing_balance_statement": transactions[-1]["balance_after_statement"] if transactions else None,
        "closing_balance_calculated": transactions[-1]["balance_after_calculated"] if transactions else None,
        "transactions": transactions
    }