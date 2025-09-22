from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from extractor import resume_keyword_extractor
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily_api = os.getenv("tavily_api")
client = TavilyClient(api_key=tavily_api)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "jobs": None})


@app.post("/upload", response_class=HTMLResponse)
async def upload_resume(request: Request, file: UploadFile):
    if not file.filename.endswith(".pdf"):
        return templates.TemplateResponse(
            "index.html", {"request": request, "jobs": [], "error": "Please upload a PDF file."}
        )

    # Save uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(file.file.read())

    # Extract keywords
    keywords = resume_keyword_extractor(file_location, top_n=20)
    keywords_list = [kw[0] for kw in keywords]
    query = "Show me jobs available relevant to: " + ", ".join(keywords_list)

    # Query Tavily
    response = client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        max_results=6
    )

    jobs = []
    for job in response.get('results', []):
        jobs.append({
            "title": job.get('title', 'N/A'),
            "url": job.get('url', '#'),
            "content": job.get('content', 'N/A')[:300] + "...",
            "score": round(job.get('score', 0), 2)
        })

    # Remove temp file
    os.remove(file_location)

    return templates.TemplateResponse("index.html", {"request": request, "jobs": jobs, "error": None})