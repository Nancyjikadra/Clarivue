from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import VideoQAPipeline
import os

from flask import Flask, render_template
app = Flask(__name__)


# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the VideoQAPipeline with the videos folder
pipeline = VideoQAPipeline(video_folder="videos") 

# Set up templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def read_contact(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
async def predict(request: Request):
    data = await request.json()
    question = data.get("question", "")
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # Use the pipeline to process the question
    result = pipeline.answer_question(question)
    
    return JSONResponse(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
