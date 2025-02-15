import os
import logging
import uvicorn
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory

from agent.config import APIConfig
from agent.tools import (
    get_weather,
    scrape_imdb,
    scrape_pdf_tabula,
    run_shell_command,
    python_repl,
    wikipedia_search,
    image_to_text,
    sql_executor,
    file_cut,
    file_copypaste,
    csv_to_json,
    md_to_html,
)

# -----------------------
# Environment Variables
# -----------------------
USE_FALLBACK_API = os.getenv("USE_FALLBACK_API", "false").lower() in ('true', '1', 'yes')
OPENWEATHERMAP_API_KEY = "b054966ea8050349af4730ced9733dec"

# -----------------------
# Configuration
# -----------------------

openai_config = APIConfig()

# -----------------------
# Logging Setup
# -----------------------

logger = logging.getLogger("agent_server")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = RotatingFileHandler("../server.log", maxBytes=10000000, backupCount=1)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# -----------------------
# Tool Configuration
# -----------------------
tools = [
    get_weather,
    scrape_imdb,
    scrape_pdf_tabula,
    run_shell_command,
    python_repl,
    wikipedia_search,
    image_to_text,
    sql_executor,
    file_cut,
    file_copypaste,
    csv_to_json,
    md_to_html,
]

llm = ChatOpenAI(
    model=openai_config.chat_model,
    openai_api_base="https://api.openai.com/v1/" if USE_FALLBACK_API else openai_config.inference_endpoint,
    openai_api_key=openai_config.auth_token,
    temperature=0.5,
    
)

# -----------------------
# Revised Memory Configuration
# -----------------------
memory = ConversationBufferMemory(memory_key="chat_history")

# -----------------------
# Enhanced Agent Initialization
# -----------------------
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    memory=memory,
    agent_kwargs={"input_variables": ["input", "agent_scratchpad", "chat_history"]},
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
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# -----------------------
# Endpoints
# -----------------------

@app.post("/run", response_class=PlainTextResponse)
async def run_task(task: str):
    if not task:
        raise HTTPException(status_code=400, detail="No task provided.")
    try:
        result = await run_in_threadpool(agent.run, task)
        return PlainTextResponse(str(result))
    except ValueError as ve:
        logger.warning(f"Task error: {task} -> {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Error executing task.")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str):
    path = Path(path).resolve()
    try:
        with open(path, "r") as file:
            return file.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Main Entry Point
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
