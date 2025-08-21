import json
from pathlib import Path
from ocr.ocr import ocr_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS
from ocr.detect_bank import detect_bank_provider
from ocr.ocr_dump import save_ocr_words_csv, save_ocr_pretty_txt
from utils import nuke_dir
from validator2 import validate_statement_json


def main():
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\n26\david n26 statements.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\downloadStatement v2.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\ptsb\ptsb ca #4018 11.01.24 - 23.09.24.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\ptsb\ptsb ca #2587 april.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\ptsb\Other 2 Pages from ptsb ca #4018 11.01.24 - 23.09.24.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\august 2025 current account account-statement_2025-01-01_2025-08-07_en-gb_423fb1-7838.pdf"
    pdf_path = r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 31 january 2025-7750.pdf"

    try:
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

    finally:
        rasters_dir = Path("results") / "ocr_rasters"
        nuke_dir(rasters_dir)

if __name__ == "__main__":
    main()
