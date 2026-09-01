from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()


from model_manager import ModelManager
from services.job_service import LinkedInJobService



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = ModelManager()
job_service = LinkedInJobService()

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
