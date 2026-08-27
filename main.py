from fastapi import FastAPI
from pydantic import BaseModel
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

    model_manager.switch(request.model)

    response = model_manager.generate(
        request.prompt
    )

    return {
        "model": request.model,
        "response": response,
    }


@app.get("/models")
def get_models():

    return model_manager.get_status()


@app.post("/models/unload")
def unload_model():

    model_manager.unload()

    return {
        "status": "Model unloaded"
    }


