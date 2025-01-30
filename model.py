import os
import numpy as np
import torch
import json
from transformers import pipeline
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import librosa

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoQAPipeline:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VideoQAPipeline, cls).__new__(cls)
        return cls._instance

    def __init__(self, video_folder="videos", cache_dir="video_cache"):
        if hasattr(self, 'initialized'):
            return
        
        try:
            logger.info("Initializing VideoQAPipeline...")
            self.initialized = True

            self.video_folder = video_folder
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)

            # Log available memory
            logger.info(f"Available memory before loading models: {torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 'CPU only'}")

            # Load models with error handling
            try:
                logger.info("Loading Whisper model...")
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.transcription_model = whisper.load_model("base", device=self.device)
                logger.info("Whisper model loaded successfully")

                logger.info("Loading SentenceTransformer model...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded successfully")

                logger.info("Loading QA pipeline...")
                self.qa_pipeline = pipeline("question-answering", 
                                         model="deepset/roberta-base-squad2", 
                                         device=0 if torch.cuda.is_available() else -1)
                logger.info("QA pipeline loaded successfully")

            except Exception as e:
                logger.error(f"Error loading models: {str(e)}")
                raise

            # Process videos
            logger.info("Processing videos...")
            self.video_transcripts = self.preprocess_all_videos()
            logger.info(f"Processed {len(self.video_transcripts)} videos")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            raise

    '''def __init__(self, video_folder="videos", cache_dir="video_cache"):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True

        self.video_folder = video_folder
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transcription_model = whisper.load_model("base", device=self.device)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)

        self.video_transcripts = self.preprocess_all_videos()'''

    def preprocess_all_videos(self):
        """Preprocess all videos in the video folder."""
        video_transcripts = {}
        for video_filename in os.listdir(self.video_folder):
            if video_filename.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(self.video_folder, video_filename)
                cache_file = os.path.join(self.cache_dir, f"{os.path.splitext(video_filename)[0]}_transcription.json")

                if os.path.exists(cache_file):
                    with open(cache_file, 'r') as f:
                        transcription_data = json.load(f)
                else:
                    transcription_data = self.process_video(video_path)
                    with open(cache_file, 'w') as f:
                        json.dump(transcription_data, f)

                # Extract full text
                full_text = transcription_data.get("text", transcription_data.get("full_text", ""))
                if not full_text:
                    full_text = " ".join([segment["text"] for segment in transcription_data["segments"]])
                
                video_transcripts[video_filename] = full_text

        return video_transcripts

    def extract_audio(self, video_path):
        """Extract audio from a video file."""
        audio_array, _ = librosa.load(video_path, sr=16000, mono=True)
        return audio_array.astype(np.float32)

    def transcribe_audio(self, audio_array):
        """Transcribe audio using Whisper."""
        return self.transcription_model.transcribe(audio_array)

    def process_video(self, video_path):
        """Extract audio and transcribe the video."""
        audio_array = self.extract_audio(video_path)
        return self.transcribe_audio(audio_array)

    def get_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        return self.embedding_model.encode(texts)

    def select_videos(self, question, top_k=2):
        """Select top-K relevant videos based on a question."""
        question_embedding = self.get_embeddings([question])
        video_transcripts_list = list(self.video_transcripts.values())
        video_names = list(self.video_transcripts.keys())

        video_embeddings = self.get_embeddings(video_transcripts_list)
        similarities = cosine_similarity(question_embedding, video_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [video_names[i] for i in top_indices]

    def answer_question(self, question):
        """Answer a single question based on selected videos."""
        # Select most relevant videos
        selected_video_names = self.select_videos(question)
        
        # Combine contexts of selected videos
        combined_context = " ".join([self.video_transcripts[video_name] for video_name in selected_video_names])

        # Get answer using QA pipeline
        answer = self.qa_pipeline(question=question, context=combined_context, max_answer_length=100)
        
        return {
            "answer": answer["answer"],
            "confidence": answer["score"],
            "context": combined_context,
            "selected_videos": selected_video_names
        }
