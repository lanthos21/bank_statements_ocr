# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based OCR system designed to extract, parse, and validate financial data from PDF bank statements. It supports multiple Irish and European banks (AIB, BOI, N26, PTSB, Revolut) and handles multi-language content (English, Spanish, French).

## Running the Application

### Main Execution
```bash
python main.py
```

The main script processes bank statement PDFs through a complete pipeline: detection → extraction → parsing → validation. Configure client PDFs by modifying the `client_pdfs` dictionary in `main.py:62`.

### Processing Strategy
The system supports three extraction strategies:
- `"auto"` (default): Native text extraction with OCR fallback
- `"native"`: Native text extraction only  
- `"ocr"`: OCR-only extraction

## Code Architecture

### Core Processing Pipeline
1. **Bank Detection** (`extract/detect_bank.py`) - Auto-identifies bank provider from PDF content using pattern matching
2. **Data Extraction** (`extract/data_extract.py`) - Extracts raw text/structured data using native PDF text or OCR (Tesseract)
3. **Bank-Specific Parsing** (`parsers/` directory) - Converts raw extracted data into structured transaction records
4. **Validation** (`validator.py`) - Validates balance consistency and transaction integrity

### Key Components

#### Bank Parser System
- `mapping.py` - Central registry mapping bank codes to OCR settings and parsers
- `parsers/` - Bank-specific parsers that implement `parse_statement(raw, client, account_type)` interface
- `extract/model_settings.py` - OCR configuration profiles per bank

#### OCR Engine
- `extract/ocr_engine.py` - Tesseract integration for image-to-text
- `extract/native_text.py` - PyMuPDF native text extraction
- Automatic fallback from native → OCR when insufficient data detected

#### Multi-Language Support
- `lang.py` - Month names and column headers in English, Spanish, French
- `utils.py` - Currency parsing, date parsing with accent handling
- `extract/detect_currency.py` - Multi-currency detection from text content

### Data Flow
```
PDF → Bank Detection → Raw Extraction → Bank Parser → Structured Statement → Validation → JSON Output
```

### Output Structure
- `results/` - Final structured JSON bundles
- `results_audit/` - Debug artifacts (raw OCR dumps, intermediate JSON)
- `results/ocr_rasters/` - Temporary image files (auto-cleaned)

### Bank Support
Each bank has dedicated parser in `parsers/`:
- `aib.py` - Allied Irish Bank
- `boi_current.py` - Bank of Ireland current accounts  
- `n26_current.py` - N26 current accounts
- `ptsb.py` - Permanent TSB
- `revolut.py` - Revolut (multi-currency support)

### Development Notes
- No formal test suite detected - validation occurs through `validator.py` balance checks
- Virtual environment located in `.venv/` (Windows-style paths)
- Debug output controlled by `DEBUG_EXTRACT = True` in `extract/data_extract.py`
- The codebase processes sensitive financial data - ensure no API keys or credentials are committed