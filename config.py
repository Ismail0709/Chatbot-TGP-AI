import io
import os
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()

gemini_client = genai.Client()

file_search_store = None
STORE_DISPLAY_NAME = "jubilee-life-store"

try:
    for store in gemini_client.file_search_stores.list():
        if store.display_name == STORE_DISPLAY_NAME:
            file_search_store = store
            print(f"Connected to existing File Search Store: {file_search_store.name}")
            break

    if not file_search_store:
        file_search_store = gemini_client.file_search_stores.create(
            config={
                "display_name": STORE_DISPLAY_NAME,
                "embedding_model": "models/gemini-embedding-2"
            }
        )
        print(f"Created new File Search Store: {file_search_store.name}")
except Exception as e:
    print(f"File Store Init Warning: {e}")

chat_sessions = {}

def get_document_page_count(file_bytes: bytes, filename: str) -> int:
    ext = filename.lower().split(".")[-1]
    if ext == "pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            return len(reader.pages)
        except Exception:
            return 1
    return 1