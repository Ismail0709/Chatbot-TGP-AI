import os
import time
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from google.genai import types
from config import gemini_client, file_search_store, get_document_page_count

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[ChatMessage]] = []


class IngestTextRequest(BaseModel):
    document_id: str
    text_content: str


class BillItem(BaseModel):
    description: str = Field(description="Item name, procedure, or service description")
    amount: Optional[str] = Field(default=None, description="Individual cost or billed amount")


class PrescriptionItem(BaseModel):
    medicine_name: str = Field(description="Name of the prescribed medicine")
    dosage_or_frequency: Optional[str] = Field(default=None, description="Dosage, quantity, or instructions")


class ClaimAttributes(BaseModel):
    document_type: Optional[str] = Field(default=None, description="inpatient or outpatient")
    employee_name: Optional[str] = Field(default=None, description="Name of the employee or patient")
    employee_claim_no: Optional[str] = Field(default=None, description="Employee claim number if visible")
    employee_contact_no: Optional[str] = Field(default=None, description="Contact number of employee/patient")
    submitted_by_employee: Optional[bool] = Field(
        default=None,
        description="True if document_type is outpatient (submitted by employee), False if inpatient"
    )
    document_summary: Optional[str] = Field(default=None, description="Short summary of the document content")
    bills_attached: bool = Field(default=False, description="True if bills or payment receipts are present")
    total_bill_amount: Optional[str] = Field(default=None, description="Total claim or bill amount listed")
    bill_summary_items: List[BillItem] = Field(default_factory=list, description="Itemized list of billing items")
    document_submitted_date: Optional[str] = Field(default=None, description="Submission date if present")
    document_received_date: Optional[str] = Field(default=None, description="Received date if present")
    prescription_attached: bool = Field(default=False, description="True if prescription documents are present")
    prescription_items: List[PrescriptionItem] = Field(default_factory=list, description="List of items/medicines prescribed")


class DocumentAnalysisResponse(BaseModel):
    is_customer_document: bool = Field(
        description="True if the document is directly from or related to a customer."
    )
    document_purpose: str = Field(
        description="Exact purpose of the document (e.g., Submission of Inpatient Claim, Submission of Outpatient Claim, Hospital Receipt, Lab Report)."
    )
    page_count: int = Field(
        description="Total page count of the document."
    )
    claim_attributes: ClaimAttributes = Field(
        description="Extracted key attributes relevant to Inpatient and Outpatient claims."
    )


@router.get("/")
def home():
    return {"status": "online", "message": "Jubilee Life RPA & AI Portal API is active."}


@router.post("/analyze/document", response_model=DocumentAnalysisResponse)
async def analyze_document(file: UploadFile = File(...)):

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded document is empty.")

        filename = file.filename or "uploaded_document.pdf"
        page_count = get_document_page_count(file_bytes, filename)
        mime_type = file.content_type or "application/pdf"

        prompt = (
            "Analyze this document for Jubilee Life RPA processing. Extract the following details strictly into the schema:\n"
            "1. Determine document_type as 'inpatient' or 'outpatient'. Set submitted_by_employee to True if outpatient, False if inpatient.\n"
            "2. Extract employee_name, employee_claim_no, employee_contact_no, document_submitted_date, and document_received_date.\n"
            "3. Provide a document_summary.\n"
            "4. Check if bills are attached (bills_attached = True/False), extract total_bill_amount, and list all itemized bill_summary_items.\n"
            "5. Check if prescriptions are attached (prescription_attached = True/False) and extract itemized prescription_items."
        )

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=DocumentAnalysisResponse,
            )
        )

        if not response.text:
            raise HTTPException(status_code=500, detail="Gemini model returned an empty response.")

        result = DocumentAnalysisResponse.model_validate_json(response.text)
        result.page_count = page_count

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/ingest/text")
def ingest_text(request: IngestTextRequest):
    if not file_search_store or not file_search_store.name:
        raise HTTPException(status_code=500, detail="File Store is not initialized.")

    temp_filename = f"{request.document_id}.txt"
    try:
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(request.text_content)

        operation = gemini_client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=file_search_store.name,
            file=temp_filename,
            config={"display_name": f"Policy Text - {request.document_id}"}
        )
        while not operation.done:
            time.sleep(1)
            operation = gemini_client.operations.get(operation)

        return {"status": "success", "message": f"Policy '{request.document_id}' successfully indexed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@router.post("/ingest/policy-document")
async def ingest_policy_document(file: UploadFile = File(...)):
    if not file_search_store or not file_search_store.name:
        raise HTTPException(status_code=500, detail="File Store is not initialized.")

    filename = file.filename or "policy_document.pdf"
    temp_path = f"temp_{filename}"
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded policy document is empty.")

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        file_ext = os.path.splitext(filename)[1].lstrip(".").lower() or "unknown"
        file_size_kb = f"{round(len(file_bytes) / 1024, 2)} KB"
        upload_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        mime_type = file.content_type or "application/octet-stream"

        auto_metadata = [
            {"key": "file_name", "string_value": filename},
            {"key": "file_extension", "string_value": file_ext},
            {"key": "file_size", "string_value": file_size_kb},
            {"key": "mime_type", "string_value": mime_type},
            {"key": "upload_timestamp", "string_value": upload_time},
        ]

        upload_config = {
            "display_name": filename,
            "custom_metadata": auto_metadata
        }

        operation = gemini_client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=file_search_store.name,
            file=temp_path,
            config=upload_config
        )
        while not operation.done:
            time.sleep(1)
            operation = gemini_client.operations.get(operation)

        return {
            "status": "success",
            "message": f"Policy document '{filename}' successfully uploaded and indexed in File Search Store.",
            "auto_metadata": auto_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload policy document: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not file_search_store or not file_search_store.name:
        raise HTTPException(status_code=500, detail="File Store is not initialized.")

    store_name: str = file_search_store.name

    gemini_history = []
    for msg in request.history:
        gemini_history.append(
            types.Content(
                role=msg.role,
                parts=[types.Part.from_text(text=msg.text)]
            )
        )

    chat = gemini_client.chats.create(
        model="gemini-3.6-flash",
        history=gemini_history,
        config=types.GenerateContentConfig(
            temperature=0.0,
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ],
            system_instruction=(
                "You are a strict RPA AI Assistant for Jubilee Life Insurance. "
                "You have access to a File Search Store containing IT/IS Policies and Claim Documents. "
                "Follow these rules strictly:\n"
                "1. STRICT GROUNDING: You MUST base your answers ONLY on the documents retrieved from the file_search tool.\n"
                "2. NO EXTERNAL KNOWLEDGE: If the user asks about topics not covered in the provided documents (e.g., general knowledge, clothing, random facts), you MUST decline to answer and state: 'I can only provide information based on Jubilee Life official documents.' Do NOT invent answers.\n"
                "3. MANDATORY CITATIONS: For every fact or conceptual explanation, you MUST append an inline citation at the end of the sentence or bullet point indicating the exact source document."
            )
        )
    )

    response = chat.send_message(request.prompt)
    bot_reply = response.text if response.text else "No response generated."

    updated_history = [
        *[msg.model_dump() for msg in request.history],
        {"role": "user", "text": request.prompt},
        {"role": "model", "text": bot_reply}
    ]

    return {
        "response": bot_reply,
        "history": updated_history
    }