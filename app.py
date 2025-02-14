import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel


from agent.config import APIConfig
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.shell import ShellTool
from langchain_community.tools.file_management import (
    CopyFileTool,
    DeleteFileTool,
    FileSearchTool,
    ListDirectoryTool,
    MoveFileTool,
    ReadFileTool,
    WriteFileTool,
)

# -----------------------
# Configuration & Logging
# -----------------------

USE_FALLBACK_API = os.getenv("USE_FALLBACK_API", "false").lower() in ('true', '1', 'yes')
openai_config = APIConfig()
WORKSPACE_PATH = Path("WORKSPACE")
WORKSPACE_PATH.mkdir(exist_ok=True)

# -----------------------
# Logging Setup
# -----------------------

logger = logging.getLogger("agent_server")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = RotatingFileHandler("server.log", maxBytes=1_000_000, backupCount=3)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# -----------------------
# Initialize LLM and Tools
# -----------------------
llm = ChatOpenAI(
    model=openai_config.chat_model,
    openai_api_base="https://api.openai.com/v1/" if USE_FALLBACK_API else openai_config.inference_endpoint,
    openai_api_key=openai_config.auth_token,
    temperature=0.5,
    max_tokens=512,
)

def get_secure_tools() -> List:
    root_dir = str(WORKSPACE_PATH.resolve())
    return [
        ShellTool(),
        ReadFileTool(root_dir=root_dir),
        WriteFileTool(root_dir=root_dir),
        ListDirectoryTool(root_dir=root_dir),
        CopyFileTool(root_dir=root_dir),
        MoveFileTool(root_dir=root_dir),
        DeleteFileTool(root_dir=root_dir),
        FileSearchTool(root_dir=root_dir),
    ]

memory = ConversationBufferMemory()
tools = get_secure_tools()

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    agent_kwargs={
        'system_message': f"You MUST operate exclusively in {WORKSPACE_PATH} directory. Never attempt to access files outside this workspace."
    }
)

# -----------------------
# FastAPI App Setup
# -----------------------
app = FastAPI(
    title="4o-Agent | TheThinkMachine",
    description="A production-ready API serving OpenAI GPT 4o-mini with file management, conversation history, and streaming support.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# -----------------------
# Pydantic Schemas
# -----------------------
class TaskRequest(BaseModel):
    task: str

class WriteFileRequest(BaseModel):
    path: str
    content: str

class SearchRequest(BaseModel):
    query: str

# -----------------------
# Endpoints
# -----------------------

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Agent API is running."}

@app.get("/config")
async def config():
    return {
        "model": openai_config.chat_model,
        "workspace": str(WORKSPACE_PATH.resolve()),
        "tools": [tool.__class__.__name__ for tool in tools],
    }

@app.get("/tools")
async def list_tools():
    return {"tools": [tool.__class__.__name__ for tool in tools]}

@app.post("/run")
async def run_task(task: TaskRequest):
    try:
        result = await run_in_threadpool(agent.run, task.task)
        logger.info(f"Task executed: {task.task} -> {result}")
        return {"result": result}
    except Exception as e:
        logger.exception("Error executing task.")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str):
    file_path = WORKSPACE_PATH / path
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with file_path.open("r") as file:
            content = file.read()
        return content
    except Exception as e:
        logger.exception("Error reading file.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/write")
async def write_file(request: WriteFileRequest):
    file_path = WORKSPACE_PATH / request.path
    try:
        with file_path.open("w") as file:
            file.write(request.content)
        logger.info(f"Wrote to file: {file_path}")
        return {"status": "success", "message": f"File {request.path} written."}
    except Exception as e:
        logger.exception("Error writing file.")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list")
async def list_directory():
    try:
        files = [str(p.relative_to(WORKSPACE_PATH)) for p in WORKSPACE_PATH.glob("**/*") if p.is_file()]
        return {"files": files}
    except Exception as e:
        logger.exception("Error listing directory.")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_files(query: str):
    try:
        # Use FileSearchTool directly for demonstration
        file_search_tool = FileSearchTool(root_dir=str(WORKSPACE_PATH.resolve()))
        result = file_search_tool.run(query)
        return {"result": result}
    except Exception as e:
        logger.exception("Error searching files.")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def conversation_history():
    try:
        # Return the raw memory buffer; in production you might sanitize or format this better.
        history = memory.buffer
        return {"history": history}
    except Exception as e:
        logger.exception("Error retrieving history.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_history():
    try:
        memory.clear()
        logger.info("Conversation memory reset.")
        return {"status": "success", "message": "Conversation history cleared."}
    except Exception as e:
        logger.exception("Error resetting history.")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Main Entry Point
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
