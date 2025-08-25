import json
from pathlib import Path
from typing import List, Dict, Any

from ocr.ocr import ocr_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS
from ocr.detect_bank import detect_bank_provider
from ocr.ocr_dump import write_ocr_dump, save_ocr_words_csv, save_ocr_pretty_txt
from utils import nuke_dir
from validator3 import validate


def _normalize_parser_output(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Make parsers interchangeable:
    - If a parser returns a bundle (has 'statements'), return that list.
    - If it returns a single statement node (has 'currencies'), wrap to a list.
    - If it returns the old single-statement-with-client shape, strip 'client'.
    """
    if not isinstance(obj, dict):
        raise ValueError("Parser output must be a dict")

    if "statements" in obj and isinstance(obj["statements"], list):
        # New bundle shape returned by parser — just pass through
        return obj["statements"]

    if "currencies" in obj and isinstance(obj["currencies"], dict):
        # Single statement node
        return [obj]

    # Old single-statement with client at top-level — convert to statement node
    if "client" in obj and "currencies" in obj:
        stmt = {k: v for k, v in obj.items() if k != "client"}
        return [stmt]

    raise ValueError("Unrecognized parser output shape")


def process_pdf(pdf_path: str, client: str) -> List[Dict[str, Any]]:
    bank_code, conf, method = detect_bank_provider(pdf_path)
    if not bank_code:
        raise RuntimeError("Could not auto-detect provider")

    print(f"🏦 Detected: {bank_code} (conf {conf:.2f}, via {method}) - {pdf_path}")

    account_type = "Current Account"  # override per file if you want

    ocr_settings = OCR_SETTINGS[bank_code]
    raw_ocr = ocr_pdf_to_raw_data(pdf_path, ocr_settings, bank_code=bank_code)

    write_ocr_dump(raw_ocr, pdf_path)

    parser_func = BANK_PARSERS[bank_code]  # MUST return a statement node OR a bundle
    parser_output = parser_func(raw_ocr, client=client, account_type=account_type)

    # Optional per-file OCR artifacts (may overwrite if reused; fine for single-file run)
    # save_ocr_words_csv(raw_ocr)
    # save_ocr_pretty_txt(raw_ocr)

    # Normalize to a list of statement nodes
    statements = _normalize_parser_output(parser_output)

    # Save each statement as its own JSON (handy for debugging)
    out_dir = Path("results_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stmt in statements:
        filename = Path(pdf_path).stem + "_structured.json"
        out_file = out_dir / filename
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(stmt, f, ensure_ascii=False, indent=2)
        # Validate statement immediately
        # validate(stmt)
        # print(f"📄 Statement saved to {out_file}")

    return statements


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
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\downloadStatement v2.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
        r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\boi may-1871.pdf",
    ]
    client = "Client 1"
    bundle = {
        "schema_version": "bank-ocr.v1",
        "client": client,
        "statements": []
    }

    try:
        # Clean out previous results_audit
        prev = Path("results_audit")
        nuke_dir(prev)

        # Process each PDF
        for pdf_path in pdf_paths:
            current_statement = process_pdf(pdf_path, client=client)
            bundle["statements"].extend(current_statement)

        # Save & validate the final bundle
        bundle_out = Path("results") / f"{client.lower().replace(' ', '_')}_bundle.json"
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
