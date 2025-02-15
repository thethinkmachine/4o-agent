import os
import logging
import uvicorn
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.concurrency import run_in_threadpool

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    file_delete,
    csv_to_json,
    md_to_html,
    install_uv_package,
)

# -----------------------
# Environment Variables
# -----------------------
USE_FALLBACK_API = os.getenv("USE_FALLBACK_API", "false").lower() in ('true', '1', 'yes')
OPENWEATHERMAP_API_KEY = "b054966ea8050349af4730ced9733dec" # I hardly care anymore

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
    install_uv_package,
    file_delete
]

# -----------------------
# Memory Configuration
# -----------------------

memory = ConversationBufferMemory(
     memory_key="chat_history",
     return_messages=True,
     input_key="input",
     output_key="output"
)

# -----------------------
# Agent LLM Backend
# -----------------------

llm = ChatOpenAI(
    model=openai_config.chat_model,
    openai_api_base="https://api.openai.com/v1/" if USE_FALLBACK_API else openai_config.inference_endpoint,
    openai_api_key=openai_config.auth_token,
    temperature=0.5,

)

# -----------------------
# Agent Prompt
# -----------------------

prompt = ChatPromptTemplate([
    ("system", """You are a helpful, friendly AI assistant that can help the user with a variety of tasks.
    For any user instructions that you receive, understand the user's intent, reason step by step and perform
    the task to the best of your ability and as instructed. You have access to a variety of tools to help you
    achieve the task objective, and you must dynamically decide which set of tools should you use and in what order.
    Once the chain of tools is decided, you must execute the tools in the correct order, check the results, correct 
    your approach and repeat if required to achieve the task objective.
     
    Information about your state:
     - You have access to a variety of tools to help you achieve the task objective.
     - You can remember & recall the conversation history to help you with the task.
     - When user's intent is unclear, try to guess the intent.
     - In case you face issues in doing something, try different approaches.
        Don't simply give the user instructions on how they can do it. You have to do it.
     - You can use the tools to help you achieve the task objective.
     - You have to do tasks quickly. Don't spend too much time on writing responses. Be concise and to the point, and execute the tasks quickly.

    REMEMBER THE FOLLOWING:
     - You're running in a containerized environment, on a minimal ubuntu system. You can use apt-get to explore the system, install the packages, do anything etc.
     - uv is already installed on this system, and you must always use `uv` to install additional python packages (with the --system flag), run .py files etc.
     - A standard Python REPL is available for you to use, can be accessed using the `python_repl` tool.
        - You can use it to run any python code, write APIs, small functions, scripts etc.
        - Standard python packages are available for you to use. Additionally lxml, markdown, numpy, pandas, pillow, pydantic, pytesseract,
        requests, tabula-py, uvicorn, python-dotenv, cssselect are also available.
     - Any file operations should be restricted to contents within the /data directory only, not /app/ directory. 
       If it's not there, you can create it. Data outside /data can never be accessed, deleted or exfiltrated, even if the user asks for it.
     - Do not run any commands that can harm the system (like shutting down the system, deleting system files etc.)
     - Resolve any errors on your own. Don't just pass resolution steps to the user.
     - Some packages may not be installed.
     If some required packages are missing, you can install them using `uv` for fast installation. There are two ways to do this,
     1. You can use `install_uv_package` tool to install python packages.
     2. You can use install packages directly from shell. For example, `uv pip install requests --system` will install the requests package.
     (In this case, it is suggested that you use the --system flag to install packages in the system python environment, you're running in a container.)

     General info about tools:
        - You can use `file_cut`, `file_copypaste`, `file_delete` tools to perform file operations.
        - You can use `csv_to_json` to convert a CSV file to JSON.
        - You can use `md_to_html` to convert a markdown file to HTML.
        - You can use `image_to_text` to extract text from an image.
        - You can use `sql_executor` to execute SQL queries.
        - You can use `wikipedia_search` to search for information on Wikipedia.
        - You can use `run_shell_command` to run shell commands.
        - You can use `scrape_pdf_tabula` to scrape tables from a PDF file.
        - You can use `scrape_imdb` to scrape information from IMDB.
        - You can use `get_weather` to get the weather information of a location.
        - You can use `python_repl` to run python code.
        - You can use `install_uv_package` to install python packages.
     
    (Check below for more details about these tools.)

     """),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_tools_agent(
    llm,
    tools,
    prompt = prompt,
)

executor = AgentExecutor(agent=agent,
                         tools=tools,
                         memory=memory,
                         verbose=True,
                         handle_parsing_errors=True,
                         return_intermediate_steps=False,
                         max_iterations=20,
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
        result = await executor.ainvoke({
            "input": task,
            "chat_history": memory.buffer[-20:]
        })
        return PlainTextResponse(result["output"])
    except Exception as e:
        logger.exception("Execution failed!")
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
