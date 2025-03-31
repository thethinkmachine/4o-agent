import os
import logging
import subprocess
import sqlite3
import tabula
import tempfile
import pandas as pd
import numpy as np
import markdown
import cssselect
import json
import httpx
import requests
import datetime
import wikipedia
import nest_asyncio
import asyncio
import playwright
import sqlalchemy
import shutil
import uvicorn
import pytesseract
import shutil
import duckduckgo_search
from lxml import html
from typing import Any, List, Dict, Optional, Union
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from langchain.tools.base import tool
from langchain_experimental.utilities import PythonREPL
from urllib.parse import urlparse
from enum import Enum
from duckduckgo_search import DDGS
from dateutil.parser import parse

# -----------------------
# Pydantic Schemas
# -----------------------

class ScrapeIMDBInput(BaseModel):
    input: Optional[str] = Field(None, description="Input (not used)")

class ScrapePDFTabulaInput(BaseModel):
    file_path: str = Field(..., description="Path to PDF file")

class RunShellCommandInput(BaseModel):
    command: str = Field(..., description="Shell command to run")

class PythonREPLInput(BaseModel):
    code: str = Field(..., description="Python code to run (in a single line or triple quotes)")

class WikipediaSearchInput(BaseModel):
    query: str = Field(..., description="Search query for Wikipedia")

class ImageToTextInput(BaseModel):
    image_path: str = Field(..., description="Path to image file")

class CSVtoJSONInput(BaseModel):
    csv_path: str = Field(..., description="Path to CSV file")
    json_path: str = Field(..., description="Output JSON path")

class SQLQueryInput(BaseModel):
    db_path: str = Field(..., description="Path to SQLite database")
    query: str = Field(..., description="SQL query to execute")

class MarkdownToHTMLInput(BaseModel):
    md_path: str = Field(..., description="Path to Markdown file")
    html_path: str = Field(..., description="Output HTML path")

class InstallUVPackageInput(BaseModel):
    package_name: str = Field(..., description="Package name or specifier")
    extra_args: str = Field("", description="Additional UV arguments")

class APICallInput(BaseModel):
    url: str = Field(..., description="Full API endpoint URL")
    method: str = Field("GET", description="HTTP method (GET, POST, PUT, DELETE)")
    headers: Optional[Dict] = Field(None, description="Request headers including auth")
    params: Optional[Dict] = Field(None, description="Query parameters")
    body: Optional[Union[Dict, str]] = Field(None, description="Request body (dict for JSON, str for raw)")
    timeout: int = Field(30, description="Timeout in seconds")

class RunPythonFileInput(BaseModel):
    code: str = Field(..., description="Python code to run.")

class ContactSortInput(BaseModel):
    input_file: str = Field(..., description="Path to the input JSON file containing contacts.")
    output_file: str = Field(..., description="Path where the sorted contacts JSON file will be written.")

class SearchType(str, Enum):
    WEB = "web"
    IMAGES = "images"
    VIDEOS = "videos"
    NEWS = "news"

class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    search_type: SearchType = Field(
        default=SearchType.WEB,
        description="Type of search (web, images, videos, news)"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return (1-10)"
    )
# -----------------------
# Define & Initialize Tools
# -----------------------
 
# SHELL COMMANDS
@tool(args_schema=RunShellCommandInput)
def run_shell_command(command: str) -> str:
    """
    Run a shell command and return the output.
    Warning: No safety checks are performed on the command.

    Args: command (str): The shell command to run
    Returns: str: The output of the command
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error ({e.returncode}): {e.stderr}"

# PYTHON UTILITIES
@tool(args_schema=InstallUVPackageInput)
def install_uv_package(package_name: str, extra_args: str = "") -> str:
    """Install Python packages using uv pip install.
    This tool is equivalent to running `uv pip install [package_name] --system`.
    
    Args: package_name (str): The package name or specifier
            extra_args (str): Additional arguments for uv

    Returns: str: Output of the installation command
    """
    command = f"uv pip install {package_name} {extra_args} --system".strip()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300  # 5-minute timeout
        )
        return f"Success:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"UV Error ({e.returncode}):\n{e.stderr}"
    except FileNotFoundError:
        return "Error: uv not found in PATH"

@tool(args_schema=PythonREPLInput)
def python_repl(code: str) -> str:
    """
    Run a single python command in a REPL environment. Not for running full scripts.
    For running full scripts, use the `run_python_file` tool.
    Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.

    Args: code (str): The Python single line/multiline commands to run, enclosed in print(...)
      E.g. "print('Hello, World!')" or "a = 5; b = 10; print(a + b)"

    Returns: str: The output of the code if print(...) is used
    """
    return PythonREPL().run(code)

@tool(args_schema=RunPythonFileInput)
def run_python_file(code: str) -> str:
    """
    Run an ephemeral Python file and return the output.
    This tool is useful for running Python code snippets or scripts in a virtual environment (not limited to REPL).

    Args: code (str): Python code to run

    Returns: str: Output of the code
    """
    tmp_filename = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
            tmp_file.write(code)
            tmp_file.flush()
            tmp_filename = tmp_file.name

        result = subprocess.run(
            ["python", tmp_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        output = result.stdout + ("\n" if result.stderr else "") + result.stderr
        return output

    except Exception as e:
        return f"Error executing code: {str(e)}"

# DATA PROCESSING UTILITIES
@tool(args_schema=ScrapePDFTabulaInput)
def scrape_pdf_tabula(file_path: str) -> str:
    """
    Scrape a PDF file using Tabula.
    
    Args: file_path (str): The absolute path to the PDF file
    Returns: str: JSON string with the scraped data
    """
    try:
        df = tabula.read_pdf(file_path, pages='all')
        return df.to_json(orient="records")
    except Exception as e:
        return f"Error scraping PDF: {e}"

@tool(args_schema=ContactSortInput)
def sort_contacts(input_file: str, output_file: str) -> None:
    """
    Read contacts from a JSON file, sort them by last name then first name,
    and write the sorted contacts back to a new JSON file.

    Args:
        input_file (str): Path to the input JSON file containing contacts.
        output_file (str): Path where the sorted contacts JSON file will be written.
    """
    try:
        # Read the contacts from the input file
        with open(input_file, 'r') as f:
            contacts = json.load(f)
        
        # Sort the contacts by last_name then first_name (case-insensitive)
        sorted_contacts = sorted(
            contacts, 
            key=lambda contact: (
                contact.get("last_name", "").lower(),
                contact.get("first_name", "").lower()
            )
        )
        
        # Write the sorted contacts to the output file with indentation for readability
        with open(output_file, 'w') as f:
            json.dump(sorted_contacts, f, indent=4)
            
        print(f"Sorted contacts have been written to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

@tool(args_schema=SQLQueryInput)
def sql_executor(db_path: str, query: str) -> List[Dict]:
    """
    Execute an SQL query on a SQLite database and return the results.

    Args: db_path (str): Path to the SQLite database
            query (str): SQL query to execute

    Returns: List[Dict]: List of dictionaries with the query results
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        results = [dict(zip([desc[0] for desc in cursor.description], row)) 
                  for row in cursor.fetchall()]
        conn.commit()
        return results
    except Exception as e:
        return {"error": str(e)}

@tool(args_schema=CSVtoJSONInput)
def csv_to_json(csv_path: str, json_path: str) -> str:
    """Convert CSV files to JSON format[2]"""
    df = pd.read_csv(csv_path)
    df.to_json(json_path, orient="records")
    return f"Converted {csv_path} to {json_path}"

@tool(args_schema=MarkdownToHTMLInput)
def md_to_html(md_path: str, html_path: str) -> str:
    """
    Convert Markdown files to HTML
    
    Args: md_path (str): Path to Markdown file
            html_path (str): Output HTML path

    Returns: str: Success message (if successfully converted) or error

    """
    with open(md_path) as f:
        html = markdown.markdown(f.read())
    with open(html_path, "w") as f:
        f.write(html)
    return f"Converted {md_path} to {html_path}"

# WEB SCRAPING & API CALLS
@tool(args_schema=APICallInput)
def make_api_call(
    url: str,
    method: str = "GET",
    headers: Optional[Dict] = None,
    params: Optional[Dict] = None,
    body: Optional[Union[Dict, str]] = None,
    timeout: int = 30
) -> Dict:
    """
    Make secure API calls with validation and error handling.
    Handles both JSON and text-based APIs.
    
    Examples:
    - GET https://api.example.com/data?param=value
    - POST https://api.example.com/submit with JSON body

    Args:
    - url (str): Full API endpoint URL
    - method (str): HTTP method (GET, POST, PUT, DELETE)
    - headers (dict): Request headers including auth
    - params (dict): Query parameters
    - body (dict or str): Request body (dict for JSON, str for raw)
    - timeout (int): Timeout in seconds (default: 30)
    
    Returns dict with:
    - status_code: HTTP status code
    - headers: Response headers
    - data: Parsed JSON or raw text
    - error: Error message if any
    """

    if not url.startswith(("http://", "https://")):
        return {"error": "Invalid URL protocol, must be http:// or https://"}
    
    method = method.upper()
    if method not in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
        return {"error": f"Invalid method {method}, must be GET, POST, PUT, DELETE, or PATCH"}

    try:
        kwargs = {
            "headers": headers or {},
            "params": params,
            "timeout": timeout
        }

        if body:
            if isinstance(body, dict):
                kwargs["json"] = body
                kwargs["headers"].setdefault("Content-Type", "application/json")
            else:
                kwargs["data"] = body

        response = requests.request(method, url, **kwargs)
    
        result = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

        try:
            result["data"] = response.json()
        except json.JSONDecodeError:
            result["data"] = response.text

        return result

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}, {response.text}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}, {response.text}"}

@tool(args_schema=DuckDuckGoSearchInput)
def duckduckgo_search(query: str, search_type: SearchType = SearchType.WEB, max_results: int = 5) -> str:
    """
    Perform a search query on DuckDuckGo and return the results.

    Args:
    - query (str): Search query string
    - search_type (SearchType): Type of search (web, images, videos, news)
    - max_results (int): Maximum number of results to return (1-10)

    Returns: str: Formatted search results or error message
    """
    try:
        max_results = min(max(1, max_results), 10)  # Clamp between 1-10
        results = []
        
        with DDGS() as ddgs:
            if search_type == SearchType.WEB:
                results = [r for r in ddgs.text(query, max_results=max_results)]
            elif search_type == SearchType.IMAGES:
                results = [r for r in ddgs.images(query, max_results=max_results)]
            elif search_type == SearchType.VIDEOS:
                results = [r for r in ddgs.videos(query, max_results=max_results)]
            elif search_type == SearchType.NEWS:
                results = [r for r in ddgs.news(query, max_results=max_results)]
            else:
                return "Invalid search type"

        if not results:
            return "No results found"
            
        # Format results
        return "\n\n".join(
            f"{i+1}. {result.get('title', 'No title')}\n"
            f"URL: {result.get('href', result.get('url', 'No URL'))}\n"
            f"Description: {result.get('body', result.get('description', 'No description'))}"
            for i, result in enumerate(results)
        )
        
    except Exception as e:
        return f"Search error: {str(e)}"

if __name__ == "__main__":
    for tool in [run_shell_command, python_repl, run_python_file, scrape_pdf_tabula, sql_executor, csv_to_json, md_to_html, make_api_call, install_uv_package]:
        print(f"Name: {tool.name}")