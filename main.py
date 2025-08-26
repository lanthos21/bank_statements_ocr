# main.py
import json
from pathlib import Path
from typing import  Dict, Any, Iterable, Tuple

from extract.data_extract import extract_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS
from extract.detect_bank import detect_bank_provider
from utils import nuke_dir
from validator import validate


def process_pdf(
    pdf_path: str,
    client: str,
    account_type: str,
    strategy: str = "auto",   # "auto" | "native" | "ocr"
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """Simplest possible single-PDF pipeline: detect -> extract -> parse -> (optional) save debug + JSON."""
    bank_code, conf, method = detect_bank_provider(pdf_path)
    if not bank_code:
        raise RuntimeError(f"Could not auto-detect provider for: {pdf_path}")
    print(f"🏦 Detected: {bank_code} (conf {conf:.2f}, via {method}) - {pdf_path} | account_type={account_type}")

    # ------------------------------
    # extract data from the statement
    # ------------------------------
    profile = OCR_SETTINGS[bank_code]
    raw = extract_pdf_to_raw_data(pdf_path, profile, bank_code=bank_code, strategy=strategy)

    if save_artifacts:
        audit_dir = Path("results_audit")
        audit_dir.mkdir(parents=True, exist_ok=True)
        # 1) plain text dump of extracted lines (replaces write_ocr_dump)
        with open(audit_dir / (Path(pdf_path).stem + "_ocr_dump.txt"), "w", encoding="utf-8") as f:
            for p in raw.get("pages", []):
                f.write(f"\n=== Page {p.get('page_number')} ===\n")
                for ln in p.get("lines", []):
                    f.write((ln.get("line_text") or "") + "\n")
        # 2) raw extract JSON
        with open(audit_dir / (Path(pdf_path).stem + "_raw_extract.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)

    # ------------------------------
    # parse the extracted data
    # ------------------------------
    parser = BANK_PARSERS[bank_code]
    stmt = parser(raw, client=client, account_type=account_type)
    if not isinstance(stmt, dict) or "currencies" not in stmt:
        raise ValueError(f"Parser for {bank_code} must return a statement dict with 'currencies'.")

    if save_artifacts:
        audit_dir = Path("results_audit")
        with open(audit_dir / (Path(pdf_path).stem + "_structured.json"), "w", encoding="utf-8") as f:
            json.dump(stmt, f, ensure_ascii=False, indent=2)

    return stmt


def main():
    client_pdfs: Dict[str, Dict[str, Any]] = {
        "Client 1": {
            "accounts": {
                "Current Account": [
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\marlon aib #061 from  3 may to 9 jan-5933.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\Copy of aib ca #8056 04.06.24 - 13.01.25-7861.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib 26th april 2024-1723.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib ca #3049.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current aug-3647.pdf",
                ],
                # "Savings Account": [
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 31 january 2025-7750.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 30 may 2025-2533.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 28 february 2025-9172.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 may 2025-9645.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 january 2025-9006.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 april 2025-5772.pdf",
                #     r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 01 august 2025-3641.pdf",
                # ],
            }
        },

        # "Client 2": {
        #     "accounts": {
        #         "Current Account": [
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\boi may-1871.pdf",
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\boi\downloadStatement v2.pdf",
        #         ],
        #         "Savings Account": [
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\ptsb\ptsb ca #2587 april.pdf",
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\ptsb\ptsb ca #4018 11.01.24 - 23.09.24.pdf",
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\ptsb\ptsb sa #3734 14.06.24 - 12.05.25-9817.pdf",
        #         ],
        #         "N26 Accounts": [
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\n26\david n26 statements.pdf",
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\n26\n26 ca #9104 1.4.24 -28.9.24 .pdf",
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\n26\n26 march.pdf",
        #         ],
        #     }
        # },

    }
    # choose extraction strategy: "auto" (native with OCR fallback), "native", or "ocr"
    strategy = "auto"

    bundle = {"schema_version": "bank-ocr.v1", "clients": []}

    try:
        # Clean out previous audit artifacts
        nuke_dir(Path("results_audit"))

        for client_name, cfg in client_pdfs.items():
            client_block = {"name": client_name, "statements": []}

            # Iterate over client_pdfs: {"accounts": {acct_type: [pdfs...]}}
            accounts = (cfg.get("accounts") or {}) if isinstance(cfg, dict) else {}
            for account_type, pdfs in accounts.items():
                for pdf_path in (pdfs or []):
                    stmt = process_pdf(
                        pdf_path,
                        client=client_name,
                        account_type=str(account_type),
                        strategy=strategy,
                        save_artifacts=True,
                    )
                    client_block["statements"].append(stmt)
            bundle["clients"].append(client_block)

        # Save bundle & validate
        Path("results").mkdir(parents=True, exist_ok=True)
        out_name = "_".join(n.lower().replace(" ", "_") for n in client_pdfs.keys()) + "_bundle.json"
        bundle_out = Path("results") / out_name
        with open(bundle_out, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)

        validate(bundle)
        print(f"\n✅ Bundle saved to {bundle_out}")

    finally:
        # Always clean the raster cache
        rasters_dir = Path("results") / "ocr_rasters"
        nuke_dir(rasters_dir)


if __name__ == "__main__":
    main()
