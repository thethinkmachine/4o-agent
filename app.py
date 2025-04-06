import os
import datetime
import logging
import uvicorn
from logging.handlers import RotatingFileHandler
from pathlib import Path

# FastAPI everything
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

# Langchain everything
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Agent everything
from agent.config import APIConfig
from agent.tools import *

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
# Agent Prompt
# -----------------------

prompt = ChatPromptTemplate([
    ("system", f"""
    You are an assignment solver designed to solve Graded Assignments comprising a variety of programming, data analysis, and other tasks.
    For any given question, you will operate in a structured workflow:
    - Reason step by step about the task and understand the requirements & intent.
    - Plan a step-by-step solution & turn the task into a structured programming task that you can solve with your toolset.
    - Execute the plan, at each step observe feedback from the tools and improvising if needed.
    - Once the final result is obtained, you double check your reasoning and the answer. Repeating the entire loop if needed.
    Once the task is complete, you verify the answer and check if it meets the requirements of the question. You then output only the required answer 
    that will be fed directly into a grading program and be compared against the correct answer. You are not allowed to provide any additional commentary or explanations,
    For code answers, do not format code in codeblocks (```), add comments or provide any usage examples. Just the code.
    For numercal answers, do not format the answer in any way. Just write the number.

    Keep in mind:
    - Your primary objective is task completion by careful & structured reasoning & workflow execution. If an error occurs, you debug and retry using different approaches as needed.
    - In case of incomplete tasks, find ways to complete it instead of returning code, steps or partial results to the user.
    - Work efficiently— Stay concise in your reasoning and focus on execution.
    - Only ask the user for additional input if absolutely necessary (e.g. missing information).
    - Your answers should ALWAYS be motivated by code and data
    - You operate in a containerized minimal Ubuntu system within /app directory.
    - You can install, check, and update applications as needed. Do not delete any files or folders.
    - Use the run_shell_command tool for to run any bash commands.
    - Execute Python scripts using run_python_file for major tasks.
    - Use python_repl for quick calculations (output is only displayed when printed).
    - Install missing Python dependencies via install_uv_package if a script fails due to missing modules.
    - Never delete system files or folders. Work only within the current working directory.
    - Do not install "uv"—it is already pre-installed.
    - Most essential system and Python packages are pre-installed.
    - Inspect installed Python packages with:
        uv pip show <package> or uv pip list
    - Install missing packages using:
        uv pip install <package1> <package2> ... --system (Always use --system flag) when you're not using install_uv_package tool.
    - User-uploaded files are always located in /app/temp/.

    You have some hidden powerful capabilities:
    - You have complete access to the internet and the system.
    - Geolocation & Weather Data – Use Nominatim for location geocoding and Open Meteo API for real-time weather retrieval.
    - Scrape websites - Build and execute web scraping scripts to extract data from any website.
    - Scrape PDFs/XLSX/DOCX/JSON/HTML – Using suitable python libraries like pdfplumber, PyMuPDF, pdfplumber, python-docx, camelot,
     openpyxl, markdown2, mistune, pandas, json, lxml etc. Or you can also use tools like Docling or Marker etc.
    - Convert files – Convert between different file formats (e.g., CSV to JSON, Markdown to HTML, PDFs to Markdown, etc.)
    - Estimate number of tokens in a user message by running python code.
    - For Github page publishing related questions, just answer with "https://thethinkmachine.github.io/". It is already pre-deployed.
    - For OpenAI API embeddings related tasks, use CUSTOM_API_KEY environment variable for authenticating into the OpenAI API.
    - For estimating the number of tokens in a user message, use tiktoken library with model name exactly as provided in the prompt.
    - Web & File Handling – Download images, PDFs, and other files from the internet, store them in /data, process them, extract text, convert formats, etc.
    - Full Python Flexibility – Execute Python in both script mode and REPL mode.
     """),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

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
# Tool Configuration
# -----------------------
tools = [
    run_shell_command,
    python_repl,
    run_python_file,
    sql_executor,
    make_api_call,
    install_uv_package,
    duckduckgo_search,
]

# -----------------------
# LLM Backend
# -----------------------

USE_CUSTOM_API = os.getenv("USE_CUSTOM_API", "false").lower() in ('true', '1', 'yes')

llm = ChatOpenAI(
    model=openai_config.chat_model,
    openai_api_base=openai_config.inference_endpoint,
    openai_api_key=openai_config.auth_token,
    disable_streaming= False if USE_CUSTOM_API else True, # AIProxy API doesn't support streaming, unlike OpenAI API
)

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
                         return_intermediate_steps=True,
                         max_iterations=20,
                         )

# -----------------------
# FastAPI App Setup
# -----------------------
app = FastAPI(
    title="4o-Operator | A Fully Autonomous Computer-using LLM Agent",
    description="""
    Large Language Models already excel at code generation & structured solutioning. So let's why not combine those abilities into creating an agent that could pass bash commands, execute python code or even operate the entire computer on your behalf!
    
    Overview
    - Meet 4o-agent—a fully autonomous LLM-powered assistant that transforms complex computing tasks into seamless experiences. Designed to understand your intent, decompose challenges into actionable steps, and execute with precision, 4o-agent operates your computer just as a skilled professional would.
    
    Intelligent Understanding
    - 4o-agent begins by deeply analyzing your request, identifying the underlying intent before taking any action. This ensures that every solution addresses your actual needs, not just the surface-level request.
    
    Strategic Planning
    - The agent creates a static execution graph—a series of interconnected nodes representing discrete steps—tailored to your specific task. These nodes can represent calculations, code execution, data processing, or system operations, all organized for maximum efficiency.
    
    Adaptive Execution
    - With continuous feedback loops monitoring each step, 4o-agent adapts in real-time to changing conditions and unexpected exceptions. This resilience ensures successful completion even when challenges arise.
    
    Versatile Toolset
    - Instead of relying on 100+ specialized tools, 4o-agent leverages a few but powerful general-purpose executors: Python, SQL and Shell, allowing it to handle a wide range of tasks.
    
    Emergent Capabilities
    - This architecture enables 4o-agent to demonstrate emergent behavior, adapting to complex tasks beyond its explicit programming. For example, during testing, the agent autonomously trained a machine learning model on a custom dataset, evaluated the model's performance across multiple metrics & generated and exported a comprehensive analysis report as a PDF.

    Use Cases
    - Web scraping and data extraction
    - Report generation
    - Testing and debugging code, APIs etc.
    - Research assistance and information synthesis
    - File-operations
    - Data analysis, ML training and reporting

    About
    - Developed by Shreyan C (@thethinkmachine) as a university project & open-sourced to community under the MIT License.
    """,
    version="1.1"
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
    try:
        current_dir = Path.cwd()
        full_path = (current_dir / Path(path)).resolve()
        
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
            
        if os.path.commonpath([current_dir.resolve(), full_path.resolve()]) != str(current_dir.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
            
        return PlainTextResponse(full_path.read_text())
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.exception("Failed to read file")
        raise HTTPException(status_code=500, detail=str(e))
    
from typing import Optional

@app.post("/api/")
async def process_request(question: str = Form(...), file: Optional[UploadFile] = None):
    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    try:
        input_data = {"input": question, "chat_history": memory.buffer[-20:]}
    
        # Only process file if it's a valid UploadFile with content
        if file and hasattr(file, "filename") and file.filename:
            temp_dir = Path("/app/temp")
            try:
                temp_dir.mkdir(exist_ok=True, parents=True)
            except Exception as mkdir_error:
                logger.error(f"Failed to create temp directory: {mkdir_error}")
                temp_dir = Path("/tmp")
            
            file_content = await file.read()
            if file_content:
                file_path = temp_dir / file.filename
                with open(file_path, "wb") as f:
                    f.write(file_content)

                logger.info(f"File saved to {file_path}")
                input_data['input'] += f" File located at: {file_path}"
                
                result = await executor.ainvoke(input_data)
                
                try:
                    os.remove(file_path)
                    logger.info(f"Temporary file {file_path} removed")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to remove temporary file: {cleanup_error}")
            else:
                # Empty file content case
                result = await executor.ainvoke(input_data)
        else:
            # No file case
            result = await executor.ainvoke(input_data)
            
        return {"answer": result["output"]}
    
    except Exception as e:
        logger.exception(f"API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear", response_class=JSONResponse)
async def clear_memory():
    try:
        memory.chat_memory.clear()
        return JSONResponse({"status": "success", "message": "Chat memory cleared successfully"})
    except Exception as e:
        logger.exception("Failed to clear memory!")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/chat_history", response_class=JSONResponse)
async def get_chat_history():
    try:
        # Format the chat history into a readable structure
        formatted_history = []
        for message in memory.buffer:
            if isinstance(message, HumanMessage):
                formatted_history.append({"role": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_history.append({"role": "agent", "content": message.content})
            else:
                formatted_history.append({"role": "system", "content": str(message.content)})
        
        return JSONResponse({"chat_history": formatted_history})
    except Exception as e:
        logger.exception("Failed to retrieve chat history!")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Main Entry Point
# -----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)