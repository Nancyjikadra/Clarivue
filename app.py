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

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    try:
        # System information
        process = psutil.Process()
        
        # Memory usage
        memory = process.memory_info()
        logger.info(f"Memory RSS: {memory.rss / 1024 / 1024:.2f} MB")
        logger.info(f"Memory VMS: {memory.vms / 1024 / 1024:.2f} MB")
        
        # CPU usage
        logger.info(f"CPU Usage: {process.cpu_percent()}%")
        
        # Available system resources
        logger.info(f"Total System Memory: {psutil.virtual_memory().total / 1024 / 1024:.2f} MB")
        logger.info(f"Available Memory: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
        
        # Python version
        logger.info(f"Python Version: {sys.version}")
        
        # Loaded modules
        logger.info("Initializing critical modules...")
        
        # Model initialization logging
        logger.info("Initializing VideoQAPipeline...")
        pipeline = VideoQAPipeline(video_folder="videos")
        logger.info("VideoQAPipeline initialized successfully")
        
        # Directory check
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Contents of videos directory: {os.listdir('videos')}")
        
        # Garbage collection
        gc.collect()
        logger.info("Garbage collection performed")
        
        logger.info("Startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}", exc_info=True)
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    try:
        # Clean up resources
        gc.collect()
        logger.info("Resources cleaned up")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

from flask import Flask, render_template
app = FastAPI(
    title="Clarivue",
    description="Ask me anything!!",
    version="1.0.0"
)
# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

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
