from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from google.genai import types
from config import gemini_client, active_db, chat_sessions, search_enterprise_documents, extract_text

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    prompt: str


class IngestTextRequest(BaseModel):
    document_id: str
    text_content: str

@router.get("/")
def home():
    return {"status": "online", "message": "Agentic RAG API is running. Go to /docs to test endpoints."}


@router.post("/ingest/text")
def ingest_text(request: IngestTextRequest):
    active_db.add_text(request.text_content, request.document_id)
    return {"status": "success", "message": f"Document {request.document_id} ingested."}


@router.post("/ingest/upload")
async def ingest_file(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

        extracted_text = extract_text(file_bytes, file.filename, file.content_type)

        if not extracted_text or not extracted_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file.")

        active_db.add_text(extracted_text, document_id=file.filename)
        return {"status": "success", "message": f"Successfully ingested {file.filename}"}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                tools=[search_enterprise_documents],
                temperature=0.2,
                system_instruction=(
                    "You are an enterprise AI agent. You have access to a local embedded document database. "
                    "If the user asks a general question, answer it directly. "
                    "If they ask about internal company data, YOU MUST use the search_enterprise_documents tool."
                )
            )
        )

    chat = chat_sessions[request.session_id]
    response = chat.send_message(request.prompt)

    return {"response": response.text}