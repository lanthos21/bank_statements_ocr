import json
from pathlib import Path
from ocr.ocr import ocr_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS

def main():
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\downloadStatement v2.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi may-1871.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut euro-9604.pdf"
    bank_code = "REVOLUT"     # could be auto-detected
    client = "Client 1"
    account_type = "Current Account"

    ocr_settings = OCR_SETTINGS[bank_code]
    raw_ocr = ocr_pdf_to_raw_data(pdf_path, ocr_settings)

    parser_func = BANK_PARSERS[bank_code]
    structured_data = parser_func(raw_ocr, client=client, account_type=account_type)

    output_file = Path("results") / (Path(pdf_path).stem + "_structured.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Structured data saved to {output_file}")

if __name__ == "__main__":
    main()
