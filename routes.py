import os
import time
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from google.genai import types
from config import gemini_client, file_search_store, chat_sessions, get_document_page_count

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    prompt: str


class IngestTextRequest(BaseModel):
    document_id: str
    text_content: str


class ClaimAttributes(BaseModel):
    patient_name: Optional[str] = Field(default=None, description="Name of the patient")
    policy_or_cnic_number: Optional[str] = Field(default=None, description="Policy number or CNIC/ID number")
    claim_type: Optional[str] = Field(default=None, description="Inpatient or Outpatient")
    hospital_or_doctor_name: Optional[str] = Field(default=None, description="Hospital, clinic, or doctor name")
    admission_date: Optional[str] = Field(default=None, description="Date of admission or visit")
    discharge_date: Optional[str] = Field(default=None, description="Date of discharge if applicable")
    claimed_amount: Optional[str] = Field(default=None, description="Total claim amount listed")
    diagnosis_or_treatment: Optional[str] = Field(default=None, description="Summary of diagnosis or treatment given")


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

        page_count = get_document_page_count(file_bytes, file.filename)
        mime_type = file.content_type or "application/pdf"

        prompt = (
            "Analyze this document for Jubilee Life RPA processing. Extract the following details:\n"
            "1. Determine if this is a document submitted by or belonging to a customer.\n"
            "2. Identify the exact purpose (e.g., Submission of Inpatient Claim, Submission of Outpatient Claim, Medical Report, Prescriptions, Bill/Receipt).\n"
            "3. Extract key attributes for Inpatient/Outpatient claims including Patient Name, Policy/CNIC, Hospital/Doctor, Admission/Discharge dates, Claimed Amount, and Diagnosis."
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
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

        result = DocumentAnalysisResponse.model_validate_json(response.text)
        result.page_count = page_count

        if file_search_store:
            temp_filename = f"temp_{file.filename}"
            try:
                with open(temp_filename, "wb") as f:
                    f.write(file_bytes)

                op = gemini_client.file_search_stores.upload_to_file_search_store(
                    file_search_store_name=file_search_store.name,
                    file=temp_filename,
                    config={"display_name": f"Claim Document - {file.filename}"}
                )
                while not op.done:
                    time.sleep(1)
                    op = gemini_client.operations.get(op)
            except Exception as store_err:
                print(f"[Warning] Failed to index document into search store: {store_err}")
            finally:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)

        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/ingest/text")
def ingest_text(request: IngestTextRequest):

    if not file_search_store:
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

    if not file_search_store:
        raise HTTPException(status_code=500, detail="File Store is not initialized.")

    temp_path = f"temp_{file.filename}"
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded policy document is empty.")

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        operation = gemini_client.file_search_stores.upload_to_file_search_store(
            file_search_store_name=file_search_store.name,
            file=temp_path,
            config={"display_name": f"Policy File - {file.filename}"}
        )
        while not operation.done:
            time.sleep(1)
            operation = gemini_client.operations.get(operation)

        return {
            "status": "success",
            "message": f"Policy document '{file.filename}' successfully uploaded and indexed in File Search Store."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload policy document: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/chat")
def chat_endpoint(request: ChatRequest):

    if not file_search_store:
        raise HTTPException(status_code=500, detail="File Store is not initialized.")

    if request.session_id not in chat_sessions:
        chat_sessions[request.session_id] = gemini_client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                temperature=0.1,
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[file_search_store.name]
                        )
                    )
                ],
                system_instruction=(
                    "You are an RPA AI Assistant for Jubilee Life Insurance. "
                    "You have access to a managed File Search Store containing: "
                    "1. Jubilee Life IT and IS (Information Security) Policies. "
                    "2. Customer Claim Documents, Medical Files, and Receipts. "
                    "When the user asks a question (e.g., policy rules or claim details), "
                    "YOU MUST use the file_search tool to find relevant information and respond strictly based on the retrieved contents."
                )
            )
        )

    chat = chat_sessions[request.session_id]
    response = chat.send_message(request.prompt)
    return {"response": response.text}