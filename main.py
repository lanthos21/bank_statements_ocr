# main.py
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple, Union

from ocr.ocr import ocr_pdf_to_raw_data
from mapping import OCR_SETTINGS, BANK_PARSERS
from ocr.detect_bank import detect_bank_provider
from ocr.ocr_dump import write_ocr_dump, save_ocr_words_csv, save_ocr_pretty_txt
from utils import nuke_dir
from validator import validate  # or: from validator import validate  (match your file name)


def process_pdf(pdf_path: str, client: str, account_type: str) -> Dict[str, Any]:
    bank_code, conf, method = detect_bank_provider(pdf_path)
    if not bank_code:
        raise RuntimeError("Could not auto-detect provider")

    print(f"🏦 Detected: {bank_code} (conf {conf:.2f}, via {method}) - {pdf_path}  |  account_type={account_type}")

    ocr_settings = OCR_SETTINGS[bank_code]
    raw_ocr = ocr_pdf_to_raw_data(pdf_path, ocr_settings, bank_code=bank_code)

    write_ocr_dump(raw_ocr, pdf_path)

    parser_func = BANK_PARSERS[bank_code]  # MUST return a statement node (dict with 'currencies')
    stmt = parser_func(raw_ocr, client=client, account_type=account_type)

    if not isinstance(stmt, dict) or "currencies" not in stmt:
        raise ValueError(f"Parser for {bank_code} must return a statement node (dict with 'currencies').")

    # Optional OCR artifacts
    # save_ocr_words_csv(raw_ocr)
    # save_ocr_pretty_txt(raw_ocr)

    # Save per-file structured JSON (handy for debugging)
    out_dir = Path("results_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (Path(pdf_path).stem + "_structured.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stmt, f, ensure_ascii=False, indent=2)

    return stmt


PDFEntry = Union[str, Tuple[str, str], Dict[str, Any]]  # str path | (path, account_type) | {"path":..., "account_type":...}

def _iter_client_entries(cfg: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """
    Yield (pdf_path, account_type) for a client's config.
    Supports either:
      - {"accounts": {"Current Account": [...], "Savings Account": [...], ...}}
      - {"account_type": "...", "pdfs": [ <PDFEntry>, ... ]}
        where each PDFEntry can be:
          * "C:\\file.pdf"                            -> uses client default account_type
          * {"path": "C:\\file.pdf", "account_type": "Savings Account"}
          * ("C:\\file.pdf", "Savings Account")
    """
    if "accounts" in cfg and isinstance(cfg["accounts"], dict):
        for acct_type, pdfs in cfg["accounts"].items():
            for p in (pdfs or []):
                yield str(p), str(acct_type)
        return

    default_acct = cfg.get("account_type", "Current Account")
    for entry in cfg.get("pdfs", []):
        if isinstance(entry, str):
            yield entry, default_acct
        elif isinstance(entry, tuple) and len(entry) == 2:
            yield str(entry[0]), str(entry[1])
        elif isinstance(entry, dict):
            path = entry.get("path")
            acct = entry.get("account_type", default_acct)
            if path:
                yield str(path), str(acct)


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
                "Savings Account": [
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 31 january 2025-7750.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 30 may 2025-2533.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 28 february 2025-9172.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 may 2025-9645.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 january 2025-9006.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 april 2025-5772.pdf",
                    r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 01 august 2025-3641.pdf",
                ],
            }
        },
        # "Client 2": {
        #     "accounts": {
        #         "Current Account": [
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 02 july 2025-3482.pdf",
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\revolut\revolut gbp.pdf",
        #         ],
        #         "Savings Account": [
        #             r"R:\DEVELOPER\FINPLAN\projects\x misc\statements\aib\aib current 31 january 2025-7750.pdf",
        #         ],
        #     }
        # },

    }

    bundle = {
        "schema_version": "bank-ocr.v1",
        "clients": []
    }

    try:
        # Clean out previous results_audit
        nuke_dir(Path("results_audit"))

        for client_name, cfg in client_pdfs.items():
            client_block = {"name": client_name, "statements": []}
            for pdf_path, account_type in _iter_client_entries(cfg):
                stmt = process_pdf(pdf_path, client=client_name, account_type=account_type)
                client_block["statements"].append(stmt)
            bundle["clients"].append(client_block)

        # Save & validate
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
