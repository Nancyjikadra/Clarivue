from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
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

# Initialize templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Global pipeline variable
pipeline = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.middleware("http")
async def monitor_memory(request: Request, call_next):
    logger.info("Memory status before request:")
    import psutil
    process = psutil.Process()
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    response = await call_next(request)
    return response

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Starting up application...")
    try:
        # Clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Initialize pipeline
        logger.info("Initializing VideoQAPipeline...")
        pipeline = VideoQAPipeline(video_folder="videos", model_size="tiny")
        logger.info("VideoQAPipeline initialized successfully")
        
        
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
    try:
        data = await request.json()
        question = data.get("question", "")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")

        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not initialized")

        result = pipeline.answer_question(question)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
