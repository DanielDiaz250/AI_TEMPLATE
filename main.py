from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

from features.chat_support.router import chat_router
from features.knowledge_base.router import kb_router

app = FastAPI(
    title="Enterprise AI Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")
api_router.include_router(chat_router)
api_router.include_router(kb_router)

app.include_router(api_router)

@app.get("/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    return {"status": "online", "modules_loaded": ["chat_support", "knowledge_base"]}