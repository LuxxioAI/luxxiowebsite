from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv
import os
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Finding variables
load_dotenv()

# --- Check for Replicate API Token ---
if not os.getenv("REPLICATE_API_TOKEN"):
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
    # Depending on your setup, you might want to exit or raise a startup error

# FastAPI app
app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str

@app.get("/")
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    input_payload = {
        "prompt": request.prompt,
        "aspect_ratio": request.aspect_ratio,
        "output_format": "png",
        "output_quality": 80,
        "safety_tolerance": 2.0,
        "prompt_upsampling": True
    }
    logger.info(f"Received generation request: prompt='{request.prompt}', aspect_ratio='{request.aspect_ratio}'")
    logger.info(f"Calling Replicate with payload: {input_payload}")

    try:
        # replicate.run() waits for completion and returns the full prediction object (dict)
        prediction_output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input_payload,
        )
        logger.info(f"Replicate call returned. Type: {type(prediction_output)}")

        # --- NEW: Process the dictionary response ---
        if not isinstance(prediction_output, dict):
            # This case should be rare if replicate.run waits, but good to check
            logger.error(f"Unexpected response type from Replicate: {type(prediction_output)}. Content: {prediction_output}")
            raise HTTPException(status_code=502, detail="Received unexpected data structure from image generation service.")

        logger.info(f"Full Replicate prediction object received: {prediction_output}") # Log the full structure

        # Check status (important!)
        status = prediction_output.get("status")
        if status == "succeeded":
            # Extract the image URL from the 'output' key
            image_result = prediction_output.get("output")

            if isinstance(image_result, str) and image_result.startswith("http"):
                # Case 1: Output is a single URL string (like the example)
                logger.info(f"Image generation successful. Image URL: {image_result}")
                return {"img": image_result}
            elif isinstance(image_result, list) and len(image_result) > 0 and isinstance(image_result[0], str) and image_result[0].startswith("http"):
                 # Case 2: Output is a list of URLs (handle just in case)
                 logger.info(f"Image generation successful. Using first Image URL from list: {image_result[0]}")
                 return {"img": image_result[0]}
            else:
                # Status is succeeded, but output format is unexpected
                logger.error(f"Replicate status is 'succeeded' but 'output' key contains unexpected data: {image_result}")
                raise HTTPException(status_code=500, detail="Image generation service succeeded but returned an invalid output format.")

        elif status in ["failed", "canceled"]:
            # Handle failure or cancellation
            error_detail = prediction_output.get("error", "No error detail provided.")
            logs = prediction_output.get("logs", "No logs provided.")
            logger.error(f"Replicate prediction failed or was canceled. Status: {status}. Error: {error_detail}. Logs: {logs}")
            # Provide a user-friendly message, potentially incorporating parts of error_detail if safe
            raise HTTPException(status_code=502, detail=f"Image generation {status}: {error_detail}")
        else:
            # Handle other unexpected statuses (e.g., starting, processing - though run should wait)
            logger.warning(f"Replicate prediction returned an unexpected status: {status}. Full response: {prediction_output}")
            raise HTTPException(status_code=502, detail=f"Image generation service returned unexpected status: {status}")

    except replicate.exceptions.ReplicateError as re:
        # Handle errors during the API call itself (network, auth, etc.)
        logger.error(f"Replicate API error during call: {re}", exc_info=False)
        raise HTTPException(status_code=502, detail=f"Error communicating with Replicate API: {str(re)}")
    except Exception as e:
        # Catch any other unexpected errors in our code
        logger.error(f"Unexpected internal server error during image generation for prompt '{request.prompt}'", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred during image generation.")

# --- Optional: Add CORS middleware if frontend and backend are on different origins ---
from fastapi.middleware.cors import CORSMiddleware

# Adjust origins as needed for your frontend development and deployment URLs
origins = [
    "http://localhost:3000",
    "http://localhost:5173", # Example for Vite
    "https://your-frontend-domain.com", # Replace with your actual deployed frontend URL
    # Add the Railway frontend URL if it's different from the backend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
