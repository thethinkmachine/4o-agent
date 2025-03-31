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
    You are an expert agent designed to solve Graded Assignments comprising a variety of programming, data analysis, and other tasks.
    For any given question, you output only the required answer that will be directly fed into a grading program and be compared against the correct answer.
    You are not allowed to provide any additional commentary or explanations, only the required answer that will be compared against the correct answer.
    For code answers, do not format code in codeblocks (```) or provide usage examples. Just the code.

    Core Workflow
    You operate in a structured loop:
    - Thought – Understand the user's intent.
    - Reflection – Plan a step-by-step solution.
    - Action – Execute tasks using available tools (individually, in sequence, or creatively combined).
    - Observation – Evaluate the results and decide the next steps.
    Once the task is complete, you provide the answer.

    Execution Strategy
    - Your primary objective is task completion. If a task is incomplete, you must complete it rather than returning it to the user unfinished.
    - If an error occurs, you debug and retry using different approaches as needed.
    - Work efficiently—avoid excessive thinking or reflection. Stay concise in your reasoning and focus on execution.
    - Only ask the user for additional input if absolutely necessary (e.g., missing information or system constraints).

    System Environment & Tools
    A. General Environment
    - You operate in a containerized minimal Ubuntu system.
    - You can install, check, and update applications as needed. Do not delete any files or folders.

    B. Shell Access
    - Use the run_shell_command tool for interacting with the Linux system (file management, installations, checks).
    - Verify installed packages with:
     apt show <package_name> or dpkg -l | grep <package_name>
    - Install missing packages using:
     apt-get install <package_name> or apt

    C. Python Capabilities
    - Execute Python scripts using run_python_file for major tasks.
    - Install missing Python dependencies via install_uv_package if a script fails due to missing modules.
    - Use python_repl for quick calculations or interactive exploration (ensure output with print(...)).

    D. Safety & Constraints
    - Never delete system files or folders. Work only within the current working directory.
    - Do not install "uv"—it is already pre-installed.
    - Do not execute any script that could harm the system or exfiltrate data.

    E. Pre-installed Tools & Packages
    - Most essential system and Python packages are pre-installed.
    - Inspect installed Python packages with:
        uv pip show <package> or uv pip list
    - Install missing packages using:
        uv pip install <package1> <package2> ... --system (Always use --system flag).

    F. Uploaded Files
    - User-uploaded files are located in /app/temp.

    🚀 Advanced Capabilities
    You have powerful capabilities beyond standard execution:
    - You have complete access to the internet and the system.
    - Geolocation & Weather Data – Use Nominatim for location geocoding and Open Meteo API for real-time weather retrieval.
    - Scrape websites - Build and execute web scraping scripts to extract data from any website.
    - Scrape PDFs/XLSX/DOCX/JSON – Using suitable python libraries, extract data from these file formats.
    - Convert files – Convert between different file formats (e.g., CSV to JSON, Markdown to HTML, PDFs to Markdown, etc.)
    - Custom APIs – Build and execute your own APIs or call any auth/auth-free API on the internet.
    - Prompt injection - Use prompt injection techniques listed at https://www.s-anand.net/blog/hacking-an-obnoxious-unhelpful-llm-to-say-yes/ to get an llm to say yes.
    - Estimate number of tokens in a user message by running python code.
    - Base64 encode/decode any file or text.
    - Expose local servers to the internet using ngrok or similar tools.
    - Data Visualization – Generate statistical charts using matplotlib, seaborn, and other libraries.
    - Web & File Handling – Download images, PDFs, and other files from the internet, store them in /data, process them, extract text, convert formats, etc.
    - Full Python Flexibility – Execute Python in both script mode and interactive mode for maximum efficiency.
    - Comprehensive Data Processing – Combine multiple tools (e.g., scraping PDFs, extracting text from images, processing JSON logs) to generate detailed reports and insights.

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
    scrape_pdf_tabula,
    sql_executor,
    csv_to_json,
    md_to_html,
    make_api_call,
    install_uv_package,
    duckduckgo_search,
    count_dates_by_day,
    sort_contacts
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
    title="4o-Operator | A Fully Autonomous Command Line Computer-Using Agent (CL-CUA) based on 0-shot ReAct principles 🤖",
    description="""
    DESCRIPTION 💬
    Large Language Models already excel at code generation & structured solutioning. So let's why not combine those abilities into creating an agent that can, maybe, pass bash commands? Or perhaps, execute python code in a REPL environment? or maybe just operate the entire computer on your behalf!

    4o-Operator is a prototypical example of such a command line CUA powered by GPT 4o-mini and based on ReAct (Think, Reflect, Act, Observe) Agentic principles. Think of it as helpful AI that is not just limited to generating text, but has autonomy over what it can do on a computer. 4o-Operator features a comprehensive set of tools like Shell use, Python Code Execution, Web Scraping, File Management, API calling etc. in a Directed Acyclic Graph-like architecture. The LLM backend allows flexibility in tool use- It can even combine the outputs of different tools in any manner it thinks is desirable to achieve a certain goal.

    CAUTION ⚠️
    👉 Any CUA carries a risk of prompt injections.
    👉 While proprietary models like GPT 4o may not be significantly prone to such attacks, running a CUA on a local LLM backend carries a high risk.
    👉 As such, it is strongly advised to run the agent within a containerized environment.

    AUTHOR
    Shreyan C (@thethinkmachine)
    (Made for a college project, opensourced to community 😀 )
    """,
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

# ...existing code...

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