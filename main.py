# ... (imports and setup remain the same) ...

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv
import os
import logging
from typing import Union, List, Any
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Env Vars ---
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
else:
    logger.info("Replicate API token found.")

# --- FastAPI App ---
app = FastAPI()

# --- Pydantic Model ---
class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str

# --- Root Endpoint ---
@app.get("/")
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {"greeting": "Hello, World!", "message": "Welcome to the Image Generation API!"}

# --- Image Generation Endpoint (FIXED) ---
@app.post("/api/generate-images")
async def generate_multiple_images(request: GenerateRequest):
    """
    Generates 3 images using the Replicate API based on the provided prompt
    and aspect ratio. Returns an array of image URLs.
    Handles FileOutput object from Replicate.
    """
    image_urls: List[str] = []
    generation_attempts = 3
    success_count = 0
    errors = []

    logger.info(f"Received request to generate {generation_attempts} images for prompt: '{request.prompt}' with aspect ratio: {request.aspect_ratio}")

    for i in range(generation_attempts):
        logger.info(f"Attempting generation {i+1}/{generation_attempts}...")
        try:
            # Call Replicate API
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",
                input={
                    "prompt": request.prompt,
                    "aspect_ratio": request.aspect_ratio,
                    "output_format": "png",
                    "output_quality": 80,
                    "safety_tolerance": 2,
                    "prompt_upsampling": True
                }
            )

            # --- Process Output (REVISED LOGIC) ---
            image_url = None
            result_item = None

            # Check if output is a list and non-empty
            if isinstance(output, list) and len(output) > 0:
                result_item = output[0] # Assume the first item holds the result
                logger.debug(f"Generation {i+1} output is list, processing first item: {type(result_item)}")
            elif output: # Check if output is not None or empty (handles direct FileOutput)
                result_item = output
                logger.debug(f"Generation {i+1} output is not list, processing directly: {type(result_item)}")
            else:
                logger.warning(f"Generation {i+1} produced empty or unexpected output: {output}")


            # If we have a result item, try to get the URL string
            if result_item:
                try:
                    # --- THE FIX: Convert result_item (potentially FileOutput) to string ---
                    image_url = str(result_item)

                    # Optional: Basic validation that it looks like a URL
                    if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
                        image_urls.append(image_url)
                        success_count += 1
                        logger.info(f"Generation {i+1} successful: {image_url}")
                    else:
                         # Log if the conversion didn't yield a valid URL string
                        error_msg = f"Generation {i+1}: Converted result is not a valid URL string. Type: {type(image_url)}, Value: {image_url}"
                        logger.warning(error_msg)
                        errors.append(error_msg)

                except Exception as convert_e:
                    # Catch errors during the str() conversion itself
                    error_msg = f"Generation {i+1}: Error converting result item to string. Original type: {type(result_item)}. Error: {convert_e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
            else:
                 # Case where output was empty or None initially
                 error_msg = f"Generation {i+1}: No valid result item found in output."
                 logger.warning(error_msg)
                 errors.append(error_msg)


        except replicate.exceptions.ReplicateError as e:
            error_msg = f"Replicate API error during generation {i+1}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during generation {i+1}: {e}"
            logger.error(error_msg, exc_info=True) # Log full traceback for unexpected errors
            errors.append(error_msg)

    # --- Final Response ---
    if not image_urls:
        logger.error(f"Failed to generate any valid images after {generation_attempts} attempts for prompt: '{request.prompt}'. Errors: {errors}")
        # Return the collected errors in the response for better debugging
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Image generation failed after {generation_attempts} attempts.",
                "errors": errors # Include the list of errors
            }
        )

    logger.info(f"Successfully generated {success_count}/{generation_attempts} images for prompt: '{request.prompt}'")

    # Return the list of successfully generated image URLs
    return {"images": image_urls}


# --- CORS Middleware (Keep as before) ---
origins = [
    "http://localhost:3000",
    "http://localhost:8081", # For Expo Go?
    # Add your frontend domains if needed
    "https://luxxio.netlify.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Run Server ---
if __name__ == "__main__":
    print("Starting FastAPI server on http://0.0.0.0:8000")
    if not REPLICATE_API_TOKEN:
       print("ERROR: REPLICATE_API_TOKEN is not set. The API will likely fail.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
