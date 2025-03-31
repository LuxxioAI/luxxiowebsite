from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv
import os
import logging
from typing import Union, List # Import Union and List for type hinting

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Finding variables
load_dotenv()

# --- Check for Replicate API Token ---
if not os.getenv("REPLICATE_API_TOKEN"):
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
    # Consider raising a startup error if critical
    # raise RuntimeError("REPLICATE_API_TOKEN environment variable not set!")

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
        "safety_tolerance": 2.0, # Adjust as needed
        "prompt_upsampling": True
    }
    logger.info(f"Received generation request: prompt='{request.prompt}', aspect_ratio='{request.aspect_ratio}'")
    logger.info(f"Calling Replicate with payload: {input_payload}")

    try:
        # replicate.run() waits for completion and returns the OUTPUT directly
        # The type hint helps clarify expected return types
        prediction_output: Union[str, List[str], None] = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input_payload,
        )
        logger.info(f"Replicate call finished. Output type: {type(prediction_output)}")
        logger.debug(f"Replicate raw output: {prediction_output}") # More detailed log

        # --- NEW: Process the direct output ---
        image_url: str | None = None

        if isinstance(prediction_output, str) and prediction_output.startswith("http"):
            # Case 1: Output is a single URL string
            image_url = prediction_output
            logger.info(f"Image generation successful. Image URL: {image_url}")

        elif isinstance(prediction_output, list):
             # Case 2: Output is a list (expecting list of URLs)
             if len(prediction_output) > 0 and isinstance(prediction_output[0], str) and prediction_output[0].startswith("http"):
                 image_url = prediction_output[0] # Take the first image
                 logger.info(f"Image generation successful (from list). Using first Image URL: {image_url}")
             else:
                 # List might be empty or contain non-URL data
                 logger.error(f"Replicate returned a list, but it was empty or did not contain a valid URL: {prediction_output}")
                 raise HTTPException(status_code=502, detail="Image generation service returned an unexpected list format.")
        else:
            # Case 3: Unexpected output type (None, dict, int, etc.)
            logger.error(f"Unexpected output type received directly from Replicate: {type(prediction_output)}. Value: {prediction_output}")
            # This could also indicate a failure reported by Replicate, though ReplicateError should ideally catch those.
            raise HTTPException(status_code=502, detail="Received unexpected data structure from image generation service.")

        # If we successfully extracted a URL
        if image_url:
             return {"img": image_url}
        else:
             # This case should technically be covered by the checks above, but as a safeguard:
             logger.error("Image URL extraction failed despite checks. This indicates a logic error.")
             raise HTTPException(status_code=500, detail="Internal error processing image generation result.")


    except replicate.exceptions.ReplicateError as re:
        # Handle errors during the API call itself (network, auth, model errors reported by Replicate)
        # These often contain useful details about why the model failed.
        error_detail = str(re)
        logger.error(f"Replicate API error during call: {error_detail}", exc_info=False) # Set exc_info=False if detail is sufficient
        # Check if the error message indicates a specific failure type (e.g., content moderation)
        if "safety handler" in error_detail.lower():
             raise HTTPException(status_code=400, detail=f"Generation failed due to safety filters: {error_detail}")
        else:
             raise HTTPException(status_code=502, detail=f"Error communicating with Replicate API: {error_detail}")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions we've deliberately raised earlier
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors in our code
        logger.error(f"Unexpected internal server error during image generation for prompt '{request.prompt}'", exc_info=True) # Log traceback for internal errors
        raise HTTPException(status_code=500, detail="An internal server error occurred during image generation.")


# --- Optional: Add CORS middleware if frontend and backend are on different origins ---
from fastapi.middleware.cors import CORSMiddleware

# Adjust origins as needed for your frontend development and deployment URLs
origins = [
    "http://localhost:3000",    # Common CRA port
    "http://localhost:5173",    # Common Vite port
    # Add your deployed frontend URL here (e.g., from Vercel, Netlify, or Railway static serving)
    # Example: "https://your-frontend-app-name.vercel.app",
    # Example: "https://luxxio-frontend.up.railway.app" # If served separately on Railway
]

# Allow your Railway backend's own URL if the frontend might be served from it too
# (though usually they have different subdomains/paths)
backend_url = os.getenv("RAILWAY_PUBLIC_DOMAIN") # Railway provides this env var
if backend_url and not backend_url.startswith("http"):
    backend_url = f"https://{backend_url}"
if backend_url and backend_url not in origins:
     origins.append(backend_url)
     logger.info(f"Added backend URL {backend_url} to CORS origins.")


# Allow all origins for local development if needed (less secure for production)
# if os.getenv("ENVIRONMENT") == "development":
#     origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the specific list
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)
