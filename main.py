from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv
import os # Import os
import logging # Import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO) # Configure basic logging
logger = logging.getLogger(__name__) # Get a logger instance

# Finding variables
load_dotenv()

# --- Check for Replicate API Token ---
# Crucial for Replicate API calls
if not os.getenv("REPLICATE_API_TOKEN"):
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
    # You might want to exit or handle this more gracefully depending on deployment
    # raise RuntimeError("REPLICATE_API_TOKEN environment variable not set!")

# FastAPI app
app = FastAPI()

class GenerateRequest(BaseModel): # Renamed class to follow Python conventions (PascalCase)
    prompt: str
    aspect_ratio: str

@app.get("/") # Changed from /home for simplicity, adjust if needed
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post("/generate") # Changed to POST to match frontend expectation and REST conventions for creating resources
async def generate_image(request: GenerateRequest): # Use Pydantic model for request body

    input_payload = {
            "prompt": request.prompt,
            "aspect_ratio": request.aspect_ratio,
            "output_format": "png",
            "output_quality": 80,
            "safety_tolerance": 2.0, # Ensure float if needed by API
            "prompt_upsampling": True
    }
    logger.info(f"Received generation request: prompt='{request.prompt}', aspect_ratio='{request.aspect_ratio}'")
    logger.info(f"Calling Replicate with payload: {input_payload}")

    try:
        # Send POST request (replace with actual request method if needed)
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro", # Make sure this model identifier is correct
            input=input_payload,
        )
        logger.info(f"Replicate call successful. Output type: {type(output)}")
        # Replicate typically returns a list of URLs or sometimes other formats
        # Log the output structure carefully upon first success
        if isinstance(output, list) and len(output) > 0:
             logger.info(f"First output element (URL expected): {str(output[0])[:150]}...") # Log beginning of output
        elif output:
             logger.info(f"Output (non-list): {str(output)[:150]}...")
        else:
             logger.warning("Replicate returned empty or unexpected output.")
             raise HTTPException(status_code=500, detail="Image generation service returned empty result.")


    except replicate.exceptions.ReplicateError as re:
        # Catch specific Replicate errors if possible (check replicate library docs for specific errors)
        logger.error(f"Replicate API error: {re}", exc_info=False) # Log replicate specific error
        # Provide a user-friendly detail, potentially hiding specifics of re
        raise HTTPException(status_code=502, detail=f"Error communicating with Replicate API: {str(re)}")
    except Exception as e:
        # Catch any other unexpected exceptions
        # Log the full traceback for detailed debugging ONLY in logs
        logger.error(f"Unexpected error during image generation for prompt '{request.prompt}'", exc_info=True)
        # Return a generic error message to the client
        raise HTTPException(status_code=500, detail=f"An internal server error occurred during image generation.") # Use str(e) for a simple message

    # Handle response (Replicate usually returns a list of image URLs)
    if not output or not isinstance(output, list) or len(output) == 0:
         logger.error(f"Unexpected output format received from Replicate: {output}")
         raise HTTPException(status_code=500, detail="Invalid response format from image generation service.")

    # Return the first image URL (assuming the frontend expects one)
    return {"img": output[0]}

# --- Optional: Add CORS middleware if frontend and backend are on different origins ---
# Needed for local development (e.g., localhost:3000 -> localhost:8000) and potentially deployment
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:8081",  # Your React dev server
    "http://localhost:8080",  # Vite dev server?
    "https://your-frontend-deployment-url.com", # Replace with your actual frontend URL
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"], # Allows all headers
)
