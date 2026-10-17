import os
import io
import uuid
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from pypdf import PdfReader
import lancedb
import pyarrow as pa
from google import genai
from google.genai import types

load_dotenv()

gemini_client = genai.Client()

class VectorStore(ABC):
    @abstractmethod
    def add_text(self, text: str, document_id: str): pass

    @abstractmethod
    def search(self, query: str) -> str: pass


class LanceDBProvider(VectorStore):
    def __init__(self, db_path="./lancedb_data"):
        self.db = lancedb.connect(db_path)
        self.table_name = "enterprise_docs"

        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 3072))
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
        else:
            self.table = self.db.open_table(self.table_name)

    def add_text(self, text: str, document_id: str):
        chunk_size = 800
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        data_to_insert = []
        for chunk in chunks:
            response = gemini_client.models.embed_content(
                model="gemini-embedding-2",
                contents=chunk
            )
            embedding = response.embeddings[0].values

            data_to_insert.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk,
                "vector": embedding
            })

        self.table.add(data_to_insert)
        print(f"Added {len(chunks)} chunks to LanceDB from {document_id}.")

    def search(self, query: str) -> str:
        query_response = gemini_client.models.embed_content(
            model="gemini-embedding-2",
            contents=query
        )
        query_embedding = query_response.embeddings[0].values

        results = self.table.search(query_embedding).limit(3).to_list()

        if results:
            return "\n\n---\n\n".join([row["content"] for row in results])
        return "No relevant documents found in the database."

active_db = LanceDBProvider()
chat_sessions = {}

def search_enterprise_documents(query: str) -> str:
    print(f"[Tool Execution] Searching LanceDB for: {query}")
    return active_db.search(query)

def extract_text(file_bytes: bytes, filename: str, mime_type: str) -> str:
    ext = filename.lower().split(".")[-1]

    if ext in ['txt', 'md', 'csv']:
        return file_bytes.decode('utf-8', errors='ignore')

    elif ext == 'pdf':
        pdf_text = ""
        try:
            pdf_reader = PdfReader(io.BytesIO(file_bytes))
            pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        except Exception as e:
            print(f"[pypdf Warning] Couldn't read digital text: {e}")

        if pdf_text and len(pdf_text.strip()) > 10:
            return pdf_text

        print(f"[Vision Fallback] '{filename}' appears to be scanned. Using Gemini Vision OCR...")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type="application/pdf"),
                "Extract all text, including handwriting, tables, and typed text from this PDF document."
            ]
        )
        return response.text

    elif ext in ['jpg', 'jpeg', 'png']:
        print(f"[Vision Pipeline] Extracting text from {filename}...")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type or "image/jpeg"),
                "Extract all text, including handwriting, tables, and receipts into plain text."
            ]
        )
        return response.text
    else:
        raise ValueError(f"Unsupported file format: {ext}")