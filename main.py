import json
from pathlib import Path
from typing import List, Dict, Any

from ocr.ocr import ocr_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS
from ocr.detect_bank import detect_bank_provider
from ocr.ocr_dump import write_ocr_dump, save_ocr_words_csv, save_ocr_pretty_txt
from utils import nuke_dir
from validator import validate


def process_pdf(pdf_path: str, client: str) -> Dict[str, Any]:
    bank_code, conf, method = detect_bank_provider(pdf_path)
    if not bank_code:
        raise RuntimeError("Could not auto-detect provider")

    print(f"🏦 Detected: {bank_code} (conf {conf:.2f}, via {method}) - {pdf_path}")

    account_type = "Current Account"  # override per file if you want

    ocr_settings = OCR_SETTINGS[bank_code]
    raw_ocr = ocr_pdf_to_raw_data(pdf_path, ocr_settings, bank_code=bank_code)

    write_ocr_dump(raw_ocr, pdf_path)

    parser_func = BANK_PARSERS[bank_code]  # MUST return a single statement node
    stmt = parser_func(raw_ocr, client=client, account_type=account_type)

    # Enforce the new contract
    if not isinstance(stmt, dict) or "currencies" not in stmt:
        raise ValueError(f"Parser for {bank_code} must return a statement node (dict with 'currencies').")

    # Optional per-file OCR artifacts
    # save_ocr_words_csv(raw_ocr)
    # save_ocr_pretty_txt(raw_ocr)

    # Save the statement as its own JSON (handy for debugging)
    out_dir = Path("results_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(pdf_path).stem + "_structured.json"
    out_file = out_dir / filename
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stmt, f, ensure_ascii=False, indent=2)

    # (Optional) per-statement validate until bundle validator is updated
    # validate(stmt)

    return stmt

def main():
    pdf_paths = [
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 31 january 2025-7750.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 30 may 2025-2533.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 28 february 2025-9172.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 may 2025-9645.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 january 2025-9006.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 april 2025-5772.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 01 august 2025-3641.pdf",
    ]
    pdf_paths = [
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut euro with pockets.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\boi may-1871.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut multi currency.pdf",
    ]
    # Provide one or two lists as appropriate
    client_pdfs: Dict[str, List[str]] = {
        "Client 1": [
            r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut euro with pockets.pdf",
            r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
        ],
        "Client 2": [
            r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\boi may-1871.pdf",
            r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut multi currency.pdf",
        ],
    }

    bundle = {
        "schema_version": "bank-ocr.v1",
        "clients": []
    }

    try:
        # Clean out previous results_audit
        prev = Path("results_audit")
        nuke_dir(prev)

        for client_name, pdf_paths in client_pdfs.items():
            client_block = {"name": client_name, "statements": []}
            for pdf_path in pdf_paths:
                stmt = process_pdf(pdf_path, client=client_name)
                client_block["statements"].append(stmt)
            bundle["clients"].append(client_block)

        # Save & (optionally) validate
        Path("results").mkdir(parents=True, exist_ok=True)
        out_name = "_".join(n.lower().replace(" ", "_") for n in client_pdfs.keys()) + "_bundle.json"
        bundle_out = Path("results") / out_name
        with open(bundle_out, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        validate(bundle)
        print(f"\n✅ Bundle saved to {bundle_out}")

    finally:
        # Always clean up raster cache
        rasters_dir = Path("results") / "ocr_rasters"
        nuke_dir(rasters_dir)


if __name__ == "__main__":
    main()
