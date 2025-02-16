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

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(..., description="The website URL to scrape. Example: 'https://example.com'."),
    headers: Optional[Dict[str, str]] = Field(None, description="Custom headers (default: User-Agent)."),
    element: Optional[str] = Field(None, description="CSS selector for specific elements."),
    attr: Optional[str] = Field(None, description="Attribute to extract (if None, returns element HTML or text).")
    

    url: str = Field(..., description="The website URL to scrape. Example: 'https://example.com'.")
    headless: bool = Field(True, description="Run the browser in headless mode. Default is True.")
    user_agent: Optional[str] = Field(
        None, description="Optional custom user agent (e.g., 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')."
    )
    tags_to_extract: Optional[List[str]] = Field(
        None, description="List of HTML tags to extract from the page. Example: ['div', 'a', 'p']."
    )

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

@tool(args_schema=ImageToTextInput)
def image_to_text(image_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args: image_path (str): Path to image file
    Returns: str: Extracted text
    """
    return pytesseract.image_to_string(Image.open(image_path))

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
    
@tool(args_schema=ScrapeWebsiteInput)
def scrape_website(url: str, headers: Optional[Dict[str, str]] = None, element: Optional[str] = None, attr: Optional[str] = None) -> Dict[str, Union[str, List[str]]]:
    """
    Scrapes a website and extracts data based on criteria. Returns complete HTML or specific element content/attributes.

    Args:
        url (str): Target URL; "http://" added if scheme missing.
        headers (dict, optional): Custom headers (default: User-Agent).
        element (str, optional): CSS selector for specific elements.
        attr (str, optional): Attribute to extract (if None, returns element HTML or text).

    Returns:
        dict: {"status": "success", "data": <extracted data>} or {"status": "error", "message": <error>}
    """
    try:
        # Ensure URL includes a scheme; add "http://" if missing.
        if not urlparse(url).scheme:
            url = "http://" + url
    
        # Default headers if not provided.
        if headers is None:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; MyScraper/1.0)"}
    
        # Perform the HTTP GET request with timeout.
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise error for bad HTTP status codes.
    
        html_content = response.text
    
        # If no element is specified, return the complete HTML of the page.
        if not element:
            return {"status": "success", "data": html_content}
    
        # Otherwise, parse the HTML and extract desired elements.
        soup = BeautifulSoup(html_content, 'html.parser')
        selected_elements = soup.select(element)
    
        if not selected_elements:
            return {"status": "error", "message": f"No elements found for selector: {element}"}
    
        if attr:
            # Return list of attribute values (if the attribute exists).
            data = [tag.get(attr) for tag in selected_elements if tag.get(attr) is not None]
            if not data:
                return {"status": "error", "message": f"Attribute '{attr}' not found in any elements matching: {element}"}
        else:
            # Return full HTML markup of each matching element.
            data = [str(tag) for tag in selected_elements]
    
        return {"status": "success", "data": data}
    
    except requests.exceptions.RequestException as req_err:
        return {"status": "error", "message": f"HTTP error occurred: {req_err}"}
    except Exception as err:
        return {"status": "error", "message": f"An unexpected error occurred: {err}"}

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
    for tool in [file_cut, file_copypaste, file_delete, run_shell_command, python_repl, run_python_file, scrape_pdf_tabula, image_to_text, sql_executor, csv_to_json, md_to_html, make_api_call, scrape_website, install_uv_package]:
        print(f"Name: {tool.name}")