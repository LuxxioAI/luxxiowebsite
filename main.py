from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv
import os
import logging
from typing import Union, List, Any # Added 'Any' for robust logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Finding variables
load_dotenv()

# --- Check for Replicate API Token ---
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
    # Depending on your setup, you might want to exit or raise a startup error
    # For FastAPI, it's often better to let it start and fail on the endpoint call

# Initialize Replicate client (optional but can be good practice)
# client = replicate.Client(api_token=REPLICATE_API_TOKEN) # If using client methods

# FastAPI app
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str

@app.get("/")
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post("/api/generate-image")
async def generate_image(prompt: str, aspect_ratio: str):
    """
    Generates an image using the Replicate API based on a prompt and aspect ratio.
    Assumes replicate.run successfully returns a URL or list of URLs.
    """

    # Directly call replicate.run - assumes it succeeds and returns the desired URL(s)
    # Removed the try/except block and the redundant/overwritten second call
    # from the original code.
    output_url = replicate.run(
        "black-forest-labs/flux-1.1-pro", # Using the first model specified in original code
        input={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "output_quality": 80,
            "safety_tolerance": 2,
            "prompt_upsampling": True
            # Note: Ensure these input parameters match the specific model's requirements
        }
    )

    # Handle response (assuming 'output_url' is the URL string or list from Replicate)
    return {"img": output_url}



        

# --- CORS Middleware (Keep as before, ensure origins are correct) ---
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    # Add your Deployed Frontend URL(s) here!
    # e.g., "https://your-frontend.on.railway.app",
    # e.g., "https://your-app.vercel.app"
]




app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
