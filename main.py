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

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    if not REPLICATE_API_TOKEN:
         logger.error("Missing REPLICATE_API_TOKEN at request time.")
         raise HTTPException(status_code=500, detail="Image generation service is not configured (missing API token).")

    input_payload = {
        "prompt": request.prompt,
        "aspect_ratio": request.aspect_ratio,
        "output_format": "png",
        "output_quality": 80,
        "safety_tolerance": 2.0, # Adjust as needed
        "prompt_upsampling": True
    }
    logger.info(f"Received generation request: prompt='{request.prompt}', aspect_ratio='{request.aspect_ratio}'")
    logger.info(f"Calling Replicate with payload: {input_payload}")

    try:
        # replicate.run() waits for completion and returns the OUTPUT directly
        prediction_output: Any = replicate.run( # Use 'Any' initially for robust debugging
            "black-forest-labs/flux-1.1-pro",
            input=input_payload,
        )

        # --- VERY IMPORTANT LOGGING ---
        output_type = type(prediction_output).__name__
        logger.info(f"Replicate call finished. Raw output type: {output_type}")
        # Log the actual value carefully, converting non-strings for logging safety
        log_output_value = str(prediction_output) if not isinstance(prediction_output, (list, dict)) else prediction_output
        # Be cautious about logging potentially large data structures in production
        logger.info(f"Replicate raw output value: {log_output_value}")


        # --- Refined Output Processing ---
        image_url: str | None = None

        # Check 1: Is it a non-empty string starting with http? (Most common success case)
        if isinstance(prediction_output, str) and prediction_output.startswith("http"):
            image_url = prediction_output
            logger.info(f"Image generation successful. Extracted URL: {image_url}")

        # Check 2: Is it a non-empty list?
        elif isinstance(prediction_output, list):
            if len(prediction_output) > 0:
                first_item = prediction_output[0]
                # Check if the first item is a string starting with http
                if isinstance(first_item, str) and first_item.startswith("http"):
                    image_url = first_item
                    logger.info(f"Image generation successful (from list). Using first URL: {image_url}")
                else:
                    # List contained unexpected data
                    first_item_type = type(first_item).__name__
                    logger.error(f"Replicate returned a list, but the first item was not a valid URL string. Got type: {first_item_type}, value: {str(first_item)[:100]}...") # Log snippet
                    raise HTTPException(status_code=502, detail="Image generation service returned a list with unexpected content.")
            else:
                # List was empty
                logger.error("Replicate returned an empty list as output.")
                raise HTTPException(status_code=502, detail="Image generation service returned an empty list.")

        # Check 3: Handle None explicitly (Could indicate Replicate issue without error)
        elif prediction_output is None:
            logger.error("Replicate returned None as output.")
            raise HTTPException(status_code=502, detail="Image generation service returned no output (None).")

        # Fallback: If none of the above matched
        else:
            logger.error(f"Replicate returned an unexpected data structure. Type: {output_type}, Value: {log_output_value}")
            # THIS is the source of the error message you were seeing.
            raise HTTPException(status_code=502, detail="Received unexpected data structure from image generation service.") # Keep original message here

        # --- Return the successful response ---
        # This part is only reached if image_url was successfully set above
        return {"img": image_url}


    except replicate.exceptions.ReplicateError as re:
        error_detail = str(re)
        logger.error(f"Replicate API error during call: {error_detail}", exc_info=False)
        # Try to give a more specific error if possible
        if "safety" in error_detail.lower():
             user_message = f"Generation failed due to safety filters: {error_detail}"
             status_code = 400 # Bad Request (user prompt issue)
        elif "timed out" in error_detail.lower():
             user_message = f"Image generation timed out: {error_detail}"
             status_code = 504 # Gateway Timeout
        else:
             user_message = f"Error communicating with Replicate API: {error_detail}"
             status_code = 502 # Bad Gateway
        raise HTTPException(status_code=status_code, detail=user_message)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions we've deliberately raised in the checks above
        logger.warning(f"Re-raising HTTPException: status={http_exc.status_code}, detail={http_exc.detail}")
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors in our *own* code
        logger.error(f"Unexpected internal server error during image generation for prompt '{request.prompt}'", exc_info=True) # Log traceback
        raise HTTPException(status_code=500, detail="An internal server error occurred during image generation.")


# --- CORS Middleware (Keep as before, ensure origins are correct) ---
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    # Add your Deployed Frontend URL(s) here!
    # e.g., "https://your-frontend.on.railway.app",
    # e.g., "https://your-app.vercel.app"
]

# Optional: Add Railway backend URL if different or needed
backend_url = os.getenv("RAILWAY_PUBLIC_DOMAIN")
if backend_url and not backend_url.startswith("http"):
    backend_url = f"https://{backend_url}"
if backend_url and backend_url not in origins:
     origins.append(backend_url)
     logger.info(f"Dynamically added backend URL {backend_url} to CORS origins.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
