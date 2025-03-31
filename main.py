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
    Generates an image using the Replicate API.
    Attempts to force the result to string at the return statement.
    """
    # Assuming replicate.run returns something that *should* represent the URL
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

    # Assign the output directly. If this 'output' is a list or other
    # non-JSON-serializable type, the error occurs at the 'return' below.
    image_url = output

    # --- Change is ONLY here ---
    # Force conversion to string directly within the return statement.
    # This might mask the underlying issue if image_url is not a string.
    try:
        return {"img": str(image_url)}
    except Exception as e:
        # Add a catch specifically around the return/conversion
        # in case str() itself fails on a very unusual type.
        print(f"Error during final conversion/return: {e}")
        print(f"Value that failed conversion: {image_url} (Type: {type(image_url)})")
        raise HTTPException(status_code=500, detail=f"Failed to format the final response: {e}")


if __name__ == "__main__":
    print("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)



        

# --- CORS Middleware (Keep as before, ensure origins are correct) ---
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://localhost:8081",
    "https://luxxio.netlify.app"
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
