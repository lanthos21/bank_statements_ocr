import json
from pathlib import Path
from ocr.ocr import ocr_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS
from ocr.detect_bank import detect_bank_provider
from ocr.ocr_dump import save_ocr_words_csv, save_ocr_pretty_txt
from validator import validate_statement_json


def main():
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut gbp.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut multi currency.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut spanish.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut euro with pockets.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\downloadStatement v2.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\boi may-1871.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut multi currency2.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\n26\n26 march.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\n26\n26 ca #9104 1.4.24 -28.9.24 .pdf"

    bank_code, conf, method = detect_bank_provider(pdf_path)
    if not bank_code:
        raise RuntimeError("Could not auto-detect provider")

    print(f"🏦 Detected: {bank_code} (conf {conf:.2f}, via {method})")

    client = "Client 1"
    account_type = "Current Account"

    ocr_settings = OCR_SETTINGS[bank_code]
    raw_ocr = ocr_pdf_to_raw_data(pdf_path, ocr_settings, bank_code=bank_code)

    # 🔹 Save raw OCR lines for debugging
    debug_txt_path = Path("results") / (Path(pdf_path).stem + "_ocr_dump.txt")
    debug_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(debug_txt_path, "w", encoding="utf-8") as f:
        for page in raw_ocr["pages"]:
            f.write(f"\n=== Page {page['page_number']} ===\n")
            for line in page["lines"]:
                f.write(line["line_text"] + "\n")
    print(f"📝 OCR debug dump saved to {debug_txt_path}")

    # 🔹 Parse into structured format
    parser_func = BANK_PARSERS[bank_code]
    structured_data = parser_func(raw_ocr, client=client, account_type=account_type)
    save_ocr_words_csv(raw_ocr)
    save_ocr_pretty_txt(raw_ocr)

    output_file = Path("results") / (Path(pdf_path).stem + "_structured.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=2)

    # Run validator against exactly what was saved
    saved_data = json.loads(Path(output_file).read_text(encoding="utf-8"))
    validate_statement_json(saved_data)

    print(f"\n\nStructured data saved to {output_file}")


if __name__ == "__main__":
    main()
