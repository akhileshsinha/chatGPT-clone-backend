from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
import shutil
import uuid
from services.document_processor import extract_document

from services.rag_service import RAGService

load_dotenv()


from model_manager import ModelManager
from services.job_service import LinkedInJobService
rag_service = RAGService()


app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
job_service = LinkedInJobService()

@app.post("/upload-document")
async def upload_document(
        file: UploadFile = File(...)
):
    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        return {
            "error": (
                f"Unsupported file type: {extension}. "
                "Supported: PDF, DOCX, XLSX, PPTX."
            )
        }

    document_id = str(uuid.uuid4())

    stored_filename = (
        f"{document_id}{extension}"
    )

    file_path = (
            UPLOAD_DIR /
            stored_filename
    )

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer,
        )

    try:
        documents = extract_document(
            str(file_path)
        )

        chunks = rag_service.chunk_documents(
            documents
        )

        rag_result = rag_service.save_document(
            document_id,
            chunks,
        )

        extracted_text = "\n\n".join(
            f"{item['source']}:\n{item['text']}"
            for item in documents
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "file_type": extension,
            "status": "processed",
            "text": extracted_text,
            "chunks": rag_result["chunk_count"],
            "embedding_dimension": (
                rag_result[
                    "embedding_dimension"
                ]
            ),
        }

    except Exception as e:
        return {
            "document_id": document_id,
            "filename": file.filename,
            "file_type": extension,
            "status": "processing_failed",
            "error": str(e),
        }
class GenerateRequest(BaseModel):
    prompt: str
    model: str = "qwen"



@app.post("/generate")
def generate(request: GenerateRequest):

    model_manager.switch("qwen")
    response = model_manager.generate(
        request.prompt
    )

    return {
        "model": "qwen",
        "response": response,
    }


@app.get("/models")
def get_models():

    return model_manager.get_status()

class SwitchModelRequest(BaseModel):
    model: str


@app.post("/models/switch")
def switch_model(request: SwitchModelRequest):

    model_manager.switch(request.model)

    return {
        "model": request.model,
        "status": "loaded",
    }

@app.post("/models/unload")
def unload_model():

    model_manager.unload()

    return {
        "status": "Model unloaded"
    }

@app.post("/generate-image")
async def generate_image(
        prompt: str = Form(...),
        image: UploadFile = File(...),
):

    model_manager.switch("qwen-vision")

    model = model_manager.models["qwen-vision"]

    image_bytes = await image.read()

    import tempfile

    with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=True
    ) as temp:

        temp.write(image_bytes)
        temp.flush()

        response = model.generate(
            temp.name,
            prompt
        )

    return {
        "model": "Qwen3-VL-2B-Instruct",
        "response": response,
    }


class GenerateCodeRequest(BaseModel):
    prompt: str


@app.post("/generate-code")
def generate_code(request: GenerateCodeRequest):

    model_manager.switch("qwen-coder")

    result = model_manager.generate(
        request.prompt
    )

    return {
        "model": "qwen-coder",
        "response": result["explanation"],
        "code": result["code"],
    }

class GenerateProjectRequest(BaseModel):
    prompt: str
    workspace: Optional[dict] = None


class WorkspaceFile(BaseModel):
    path: str
    content: str

class Workspace(BaseModel):
    files: list[WorkspaceFile] = []


@app.post("/generate-project")
def generate_project(request: GenerateProjectRequest):

    model_manager.switch("qwen-coder")

    workspace_files = (
        request.workspace.get("files", [])
        if request.workspace
        else []
    )

    response = model_manager.generate_project(
        request.prompt,
        workspace_files,
    )

    return response


class AskDocumentRequest(BaseModel):
    document_id: str
    question: str

@app.post("/ask-document")
def ask_document(
        request: AskDocumentRequest
):
    model_manager.switch("qwen")

    results = rag_service.search(
        request.document_id,
        request.question,
        top_k=5,
    )

    if not results:
        return {
            "document_id": request.document_id,
            "question": request.question,
            "answer": (
                "I could not find the answer "
                "in the document."
            ),
            "sources": [],
        }

    context_parts = []

    for index, result in enumerate(
            results,
            start=1,
    ):
        context_parts.append(
            f"""
Source {index} ({result['source']}):
{result['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
Answer the user's question using only the
provided document context.

Rules:
- Do not use outside knowledge.
- If the answer is not available in the context,
  say exactly:
  "I could not find the answer in the document."
- Give a concise and accurate answer.
- When useful, mention the source such as
  Page 3, Slide 5, or a sheet name.
- Do not invent information.

Document context:
{context}

User question:
{request.question}
"""

    response = model_manager.generate(
        prompt
    )

    return {
        "document_id": request.document_id,
        "question": request.question,
        "answer": response,
        "sources": results,
    }