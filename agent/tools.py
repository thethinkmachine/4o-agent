import os
import logging
import subprocess
import sqlite3
import tabula
import pandas as pd
import numpy as np
import markdown
import cssselect
import json
import httpx
import requests
from lxml import html
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
import pytesseract
import shutil
from pydantic import BaseModel, Field
from langchain.tools.base import tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

OPENWEATHERMAP_API_KEY = "b054966ea8050349af4730ced9733dec"
# Yeah I know, I'm not supposed to do this, but don't worry it's not tied to my bank account or anything

# -----------------------
# Pydantic Schemas
# -----------------------
class FileCutInput(BaseModel):
    src: str = Field(..., description="Source file path")
    dest: str = Field(..., description="Destination directory")

class FileCopyPasteInput(BaseModel):
    src: str = Field(..., description="Source file path or directory")
    dest: str = Field(..., description="Destination directory")

class FileDeleteInput(BaseModel):
    path: str = Field(..., description="File path to delete")

class GetWeatherInput(BaseModel):
    location: str = Field(..., description="Location (city name, country code (2-letter alphabetical))")

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

# -----------------------
# Define & Initialize Tools
# -----------------------

# FILE MANAGEMENT

@tool(args_schema=FileCutInput)
def file_cut(src: str, dest: str) -> str:
    """
    Move files between directories
    
    Args: src (str): Source file path (file or directory)
            dest (str): Destination directory path
    
    Returns: str: Success message (if successfully moved) or error
    """
    try:
        shutil.move(src, dest)
        return f"Moved {src} to {dest}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool(args_schema=FileCopyPasteInput)
def file_copypaste(src: str, dest: str) -> str:
    """
    Copy-paste files or directories from source to destination

    Args: src (str): Source file path or directory
            dest (str): Destination directory path
    
    Returns: str: Success message (if successfully pasted) or error
    """
    try:
        if os.path.isfile(src):
            shutil.copy2(src, dest)  # For files, copy
            return f"Pasted (copied) {src} to {dest}"
        elif os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, os.path.basename(src)))  # For directories, copy the entire tree
            return f"Pasted (copied) directory {src} to {dest}"
        else:
            return "Source not found"
    except Exception as e:
        return f"Error: {str(e)}"


@tool(args_schema=FileDeleteInput)
def file_delete(path: str) -> str:
    """Delete files with existence check[5]"""
    if not os.path.exists(path):
        return "File not found"
    try:
        os.remove(path)
        return f"Deleted {path}"
    except Exception as e:
        return f"Error: {str(e)}"
    
# SHELL & PYTHON UTILITIES
    
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
        return f"Error executing command: {e}"
    
@tool(args_schema=InstallUVPackageInput)
def install_uv_package(package_name: str, extra_args: str = "") -> str:
    """Install Python packages using uv.
    
    Args: package_name (str): The package name or specifier
            extra_args (str): Additional arguments for uv

    Returns: str: Output of the installation command
    """
    command = f"uv pip install {package_name} {extra_args}".strip()
    
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
    Run Python code in a REPL shell and return the output.
    Use it for quick Python code execution involving math, data processing,
    quick python functions, etc. Do not use for long-running tasks.


    Args: code (str): The Python code to run. \
        Enclose in triple quotes or use semi-colons for multi-line code.
    Returns: str: The output of the code (ONLY when using print statements)
    """
    return PythonREPL().run(code)
    
# INTERNET UTILITIES

@tool(args_schema=GetWeatherInput)
def get_weather(location: str) -> str:
    """
    Get the current weather for a location using the OpenWeatherMap API.
    
    Args: location (str): The location (city's name, comma, 2-letter alphabetical country code (ISO3166))
    Returns: JSON Object with weather data
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHERMAP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return f"Error fetching weather data: {response.text}"
    
@tool(args_schema=WikipediaSearchInput)
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for a query and return the top 3 results.

    Args: query (str): The search query
    Returns: str: The top 3 search results from Wikipedia
    """
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3)).run(query)

# DATA PROCESSING UTILITIES

@tool(args_schema=ScrapeIMDBInput)
def scrape_imdb(input: any) -> List[Dict[str, str]]:
    """Fetch the top boxoffice movies from IMDb. 
    
    Args: input (any) (not used)
    Returns: List[Dict[str, str]]: List of movie dictionaries with title, year, and rating
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; IMDbBot/1.0)"}
    response = httpx.get("https://www.imdb.com/chart/top/", headers=headers)
    response.raise_for_status()
    tree = html.fromstring(response.text)
    movies = []
    for item in tree.cssselect(".ipc-metadata-list-summary-item"):
        title = (item.cssselect(".ipc-title__text")[0].text_content() if item.cssselect(".ipc-title__text") else None)
        year = (item.cssselect(".cli-title-metadata span")[0].text_content() if item.cssselect(".cli-title-metadata span") else None)
        rating = (item.cssselect(".ipc-rating-star")[0].text_content() if item.cssselect(".ipc-rating-star") else None)
        if title and year and rating:
            movies.append({"title": title, "year": year, "rating": rating})
    return movies

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