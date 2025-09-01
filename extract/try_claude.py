import json
import base64
import requests
from typing import Dict, Any, Optional
import logging


class ClaudeBankStatementParser:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }

    def pdf_to_base64(self, pdf_path: str) -> str:
        """Convert PDF file to base64 string"""
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')

    def create_claude_prompt(self, client_name: str = None) -> str:
        """Create the prompt for Claude to return just a statement object"""
        return f"""
You are a bank statement parser that extracts transaction data from PDF bank statements and returns a structured JSON statement object.

CRITICAL: Your entire response MUST be a single, valid JSON object matching the exact schema below. DO NOT include any text outside of the JSON structure, including backticks, explanations, or markdown.

Return ONLY a statement object in this exact format:
{{
  "statement_id": "unique_identifier", // use format: institution_lastname_startdate_enddate
  "file_name": "extracted_filename_or_generated",
  "institution": "bank_name",
  "account_type": "Current Account", // or Savings, Credit Card, etc.
  "iban": "account_number_or_iban",
  "statement_start_date": "YYYY-MM-DD",
  "statement_end_date": "YYYY-MM-DD",
  "currencies": {{
    "EUR": {{ // or USD, GBP, etc. - use appropriate currency from statement
      "balances": {{
        "opening_balance": {{
          "summary_table": null, // amount from summary section if visible
          "transactions_table": null // amount from first transaction if available
        }},
        "money_in_total": {{
          "summary_table": null, // from summary if available
          "transactions_table": 0.00 // calculated sum of all credits
        }},
        "money_out_total": {{
          "summary_table": null, // from summary if available
          "transactions_table": 0.00 // calculated sum of all debits
        }},
        "closing_balance": {{
          "summary_table": null, // from summary if available
          "transactions_table": null,
          "calculated": 0.00 // opening + money_in - money_out
        }}
      }},
      "transactions": [
        {{
          "seq": 0, // sequential number starting from 0
          "transaction_date": "YYYY-MM-DD",
          "transaction_type": "debit", // or "credit"
          "description": "transaction description", // clean up OCR errors
          "amount": 0.00, // always positive number
          "signed_amount": 0.00 // negative for debits, positive for credits
        }}
        // ... continue for ALL transactions found
      ]
    }}
  }}
}}

Parsing Instructions:
1. Extract ALL visible transactions from the statement
2. transaction_type: "debit" for money out, "credit" for money in
3. Clean up obvious OCR errors in descriptions but keep original meaning
4. amount: always positive numbers
5. signed_amount: negative for debits, positive for credits
6. Calculate totals from transaction data
7. Extract client name from statement if provided client_name is null
8. Use null for fields that cannot be determined from the PDF
9. Dates must be in YYYY-MM-DD format

Client name for this statement: {client_name or "extract from PDF"}

RESPOND ONLY WITH THE STATEMENT JSON OBJECT. NO OTHER TEXT.
"""

    def parse_statement_with_claude(self, pdf_path: str, client_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Send PDF to Claude API for parsing

        Args:
            pdf_path: Path to the PDF file
            client_name: Optional client name to include in prompt

        Returns:
            Statement object (not full JSON structure) or None if parsing failed
        """
        try:
            # Convert PDF to base64
            pdf_base64 = self.pdf_to_base64(pdf_path)

            # Create the API request
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 8000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": self.create_claude_prompt(client_name)
                            }
                        ]
                    }
                ]
            }

            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )

            if not response.ok:
                logging.error(f"Claude API error: {response.status_code} - {response.text}")
                return None

            response_data = response.json()
            claude_response = response_data["content"][0]["text"]

            # Parse Claude's JSON response
            try:
                statement_data = json.loads(claude_response)
                return statement_data
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Claude's JSON response: {e}")
                logging.error(f"Claude response: {claude_response}")
                return None

        except Exception as e:
            logging.error(f"Error parsing statement with Claude: {e}")
            return None

    def merge_statement_into_existing_json(self, existing_json: Dict[str, Any], statement_data: Dict[str, Any],
                                           client_name: str) -> Dict[str, Any]:
        """
        Merge Claude-parsed statement into existing JSON structure

        Args:
            existing_json: Your existing in-memory JSON with known bank statements
            statement_data: Claude's parsed statement object
            client_name: Name of the client this statement belongs to

        Returns:
            Merged JSON structure
        """
        if not statement_data:
            return existing_json

        # Find existing client or create new one
        existing_client = None
        for client in existing_json["clients"]:
            if client["name"] == client_name:
                existing_client = client
                break

        if existing_client:
            # Add statement to existing client
            existing_client["statements"].append(statement_data)
        else:
            # Create new client
            existing_json["clients"].append({
                "name": client_name,
                "statements": [statement_data]
            })

        return existing_json


def parse_unknown_statement(pdf_path: str, api_key: str, client_name: str) -> Optional[Dict[str, Any]]:
    """
    Simple function to parse a single unknown PDF statement with Claude

    Args:
        pdf_path: Path to the PDF that your Python parser couldn't handle
        api_key: Your Anthropic API key
        client_name: Name of the client this statement belongs to

    Returns:
        Statement object ready to be inserted into your main JSON, or None if parsing failed
    """
    parser = ClaudeBankStatementParser(api_key)
    statement_data = parser.parse_statement_with_claude(pdf_path, client_name)
    return statement_data


def integrate_claude_parser(existing_json_path: str, unknown_pdf_path: str, api_key: str, client_name: str = None) -> \
Dict[str, Any]:
    """
    DEPRECATED: Use parse_unknown_statement() instead for in-memory workflows

    Main integration function that combines your existing parsing with Claude fallback

    Args:
        existing_json_path: Path to your existing parsed JSON file
        unknown_pdf_path: Path to the PDF that your Python parser couldn't handle
        api_key: Your Anthropic API key
        client_name: Optional client name if known

    Returns:
        Combined JSON structure
    """
    # Load existing JSON
    with open(existing_json_path, 'r', encoding='utf-8') as f:
        existing_json = json.load(f)

    # Parse unknown statement with Claude
    parser = ClaudeBankStatementParser(api_key)
    statement_data = parser.parse_statement_with_claude(unknown_pdf_path, client_name)

    if statement_data:
        # Merge results
        combined_json = parser.merge_statement_into_existing_json(existing_json, statement_data,
                                                                  client_name or "Unknown Client")
        return combined_json
    else:
        logging.error("Claude parsing failed - returning original JSON")
        return existing_json


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Your API key (store securely in environment variable)
    API_KEY = "your_anthropic_api_key_here"

    # Example: Parse a single unknown statement
    statement_data = parse_unknown_statement(
        pdf_path="unknown_statement.pdf",
        api_key=API_KEY,
        client_name="John Doe"
    )

    if statement_data:
        print("Successfully parsed statement!")
        print(json.dumps(statement_data, indent=2))

        # In your main code, you would do:
        # your_in_memory_json["clients"][client_index]["statements"].append(statement_data)
    else:
        print("Failed to parse statement")