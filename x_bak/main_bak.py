import json
from pathlib import Path
from ocr import ocr_pdf_to_raw_data
from parsers.boi_current import parse_statement
from parsers.revolut import parse_statement


BANK_PARSERS = {
    "BOI": parse_statement,
    "REVOLUT": parse_statement,
}

def main():
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\downloadStatement v2.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi may-1871.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut euro-9604.pdf"

    client = "Client 1"
    account_type = "Current Account"
    bank_code = "REVOLUT"  # This could be auto-detected in the future

    # Stage 1 - Generic OCR
    raw_ocr = ocr_pdf_to_raw_data(pdf_path)

    # Stage 2 - Bank-specific parser
    parser_func = BANK_PARSERS.get(bank_code)
    if not parser_func:
        raise ValueError(f"No parser found for bank code {bank_code}")

    structured_data = parser_func(raw_ocr, client=client, account_type=account_type)

    # Save final JSON
    output_file = Path("../results") / (Path(pdf_path).stem + "_structured.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Structured data saved to {output_file}")

if __name__ == "__main__":
    main()
