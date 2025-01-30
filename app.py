from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model import VideoQAPipeline
import os

import logging
import psutil
import sys
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global pipeline variable
pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Starting up application...")
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize pipeline
        logger.info("Initializing VideoQAPipeline...")
        pipeline = VideoQAPipeline(video_folder="videos")
        logger.info("VideoQAPipeline initialized successfully")
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise e
        
@app.on_event("shutdown")
async def shutdown_event():
    global pipeline
    logger.info("Shutting down application...")
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear pipeline
        pipeline = None
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleanup completed")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

from flask import Flask, render_template
app = FastAPI(
    title="Clarivue",
    description="Ask me anything!!",
    version="1.0.0"
)


# Initialize the VideoQAPipeline with the videos folder
try:
    pipeline = VideoQAPipeline(video_folder="videos")
except Exception as e:
    print(f"Error initializing pipeline: {e}")

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
