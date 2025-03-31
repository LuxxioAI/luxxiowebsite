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
    Generates an image using Replicate and returns the image URL.
    Assumes replicate.run returns a string URL or a list containing one URL string.
    """
    logger.info(f"Received image generation request: prompt='{prompt}', aspect_ratio='{aspect_ratio}'")

    # Call Replicate API
    # If this call fails (network, API key, Replicate error), FastAPI will
    # return a 500 Internal Server Error by default, as error catching was removed.
    output = replicate.run(
        "black-forest-labs/flux-1.1-pro",
        input={
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "output_quality": 80,
            "safety_tolerance": 2,
            "prompt_upsampling": True
        }
    )

    logger.info(f"Replicate output received (type: {type(output)}): {output}")

    # --- Robust handling based on previous debugging ---
    # Handle potential list output from Replicate cleanly
    final_url = None
    if isinstance(output, list):
        if output and isinstance(output[0], str):
             final_url = output[0] # Extract URL if it's a list of strings
        else:
            logger.error(f"Replicate returned a list, but it was empty or first item wasn't a string: {output}")
    elif isinstance(output, str):
        final_url = output # Assume it's the URL string directly
    else:
        logger.error(f"Replicate returned an unexpected data type: {type(output)}")

    # If we still don't have a valid string URL, raise an error
    if not isinstance(final_url, str):
         logger.error(f"Failed to extract a valid string URL from Replicate response: {output}")
         # This HTTPException is for *processing* the response, not the call itself
         raise HTTPException(
             status_code=500,
             detail="Image generated, but couldn't process the response format from Replicate."
         )
    # --- End robust handling ---

    logger.info(f"Returning image URL: {final_url}")

    # Return the result directly assuming 'final_url' is now a string URL
    return {"img": final_url}



        

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
