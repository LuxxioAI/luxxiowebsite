# main.py
import os
import base64
import json
import tempfile
import stripe # type: ignore
from fastapi import FastAPI, HTTPException, Body, Request, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, Optional
from google.cloud import firestore
from google.cloud import storage
from PIL import Image
import requests
import io
import uuid
from google.cloud.firestore_v1.base_query import FieldFilter
from pydantic import BaseModel, Field
import replicate
import logging
import uvicorn
import datetime

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()

# --- START OF MODIFIED SECTION (WITH ENHANCED DEBUGGING) ---

def setup_google_credentials():
    """
    Sets up Google Cloud credentials with enhanced debugging.
    Priority 1 (Local Dev): Looks for 'serviceAccountKey.json' in the root directory.
    Priority 2 (Deployment): Falls back to the Base64 encoded environment variable.
    """
    local_key_path = "serviceAccountKey.json"

    # --- Enhanced Debugging Logs ---
    current_working_directory = os.getcwd()
    absolute_file_path = os.path.abspath(local_key_path)
    
    logger.info("--- [Google Credential Setup Debug] ---")
    logger.info(f"Current Working Directory: {current_working_directory}")
    logger.info(f"Checking for credentials file at absolute path: {absolute_file_path}")
    try:
        directory_contents = os.listdir(current_working_directory)
        logger.info(f"Files/Folders in current directory: {directory_contents}")
    except Exception as e:
        logger.warning(f"Could not list directory contents: {e}")
    logger.info("---------------------------------------")
    # --- End Enhanced Debugging ---

    # Priority 1: Check for local service account file for development
    if os.path.exists(local_key_path):
        logger.info(f"SUCCESS: Found '{local_key_path}'. Using it for local development credentials.")
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = local_key_path
        return local_key_path
    else:
        logger.warning(f"INFO: Local file '{local_key_path}' was NOT found in the working directory.")

    # Priority 2: Check for Base64 encoded environment variable for deployment
    base64_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if base64_key:
        try:
            logger.info("Attempting to set up Google credentials from Base64 environment variable (for deployment)...")
            key_data = base64.b64decode(base64_key).decode('utf-8')
            key_json = json.loads(key_data)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(key_json, temp_file)
                temp_file_path = temp_file.name

            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
            logger.info(f"SUCCESS: Google credentials file created from Base64 at: {temp_file_path}")
            return temp_file_path
        except Exception as e:
            logger.error(f"ERROR: Failed to setup Google credentials from Base64: {e}")
            return None

    # If neither method works
    logger.error("FATAL: No valid Google credentials found. Neither 'serviceAccountKey.json' was found locally nor was 'GOOGLE_APPLICATION_CREDENTIALS_BASE64' set correctly.")
    return None

# --- Configuration using the setup function ---
credentials_path = setup_google_credentials()

# Initialize clients (they will be None if credentials failed)
db = None
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION_NAME", "shipments")
storage_client = None
bucket = None
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")

if credentials_path:
    try:
        db = firestore.AsyncClient()
        logger.info("✅ Firestore client initialized successfully.")
        logger.info(f"Using Firestore collection: {FIRESTORE_COLLECTION}")
    except Exception as e:
        logger.error(f"🔴 Failed to initialize Firestore client: {e}")
        db = None
    
    if FIREBASE_STORAGE_BUCKET:
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(FIREBASE_STORAGE_BUCKET)
            logger.info(f"✅ Firebase Storage client initialized for bucket: {FIREBASE_STORAGE_BUCKET}")
        except Exception as e:
            logger.error(f"🔴 Failed to initialize Firebase Storage client: {e}")
            bucket = None
    else:
        logger.warning("⚠️ FIREBASE_STORAGE_BUCKET environment variable not set. Firebase Storage will be disabled.")
else:
    logger.error("🔴 Google credentials setup failed. Firestore and Storage clients will not be available.")

# --- END OF MODIFIED SECTION ---


# --- Stripe Configuration ---
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
if not STRIPE_SECRET_KEY:
    logger.error("🔴 FATAL ERROR: Stripe Secret Key not found in .env file.")
else:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe API Key Loaded.")

# --- Replicate Configuration ---
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
else:
    logger.info("✅ Replicate API token found.")

# --- FastAPI App ---
app = FastAPI(
    title="Image Generation & E-commerce API",
    description="API for image generation, payment processing, and shipment tracking",
    version="1.0.0"
)

# --- Pydantic Models ---
class GenerateRequest(BaseModel):
    prompt: str
    aspect_ratio: str

class ShipmentData(BaseModel):
    reference: str = Field(..., example="test001")
    orderid: str = Field(default="", example="")
    shipment_company: str = Field(..., example="DPD")
    tracking_code: str = Field(..., example="05132088424538")
    tracking_url: str = Field(..., example="https://www.dpdgroup.com/nl/mydpd/my-parcels/search?lang=nl")

# --- Helper Function ---
def calculate_order_amount_cents(total_price_euros: float) -> int:
    if total_price_euros <= 0:
        raise ValueError("Total price must be positive.")
    return int(round(total_price_euros * 100))

# --- Root Endpoint ---
@app.get("/")
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {
        "greeting": "Hello, World!", 
        "message": "Welcome to the Image Generation API!",
        "firestore_status": "connected" if db else "disconnected",
        "storage_status": "connected" if bucket else "disconnected",
        "stripe_status": "configured" if STRIPE_SECRET_KEY else "not configured",
        "replicate_status": "configured" if REPLICATE_API_TOKEN else "not configured"
    }

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "services": {
            "firestore": "connected" if db else "disconnected",
            "storage": "connected" if bucket else "disconnected",
            "stripe": "configured" if STRIPE_SECRET_KEY else "not configured",
            "replicate": "configured" if REPLICATE_API_TOKEN else "not configured"
        }
    }

# --- Firebase Storage Upload Helper ---
def upload_to_firebase_storage(file_bytes: bytes, destination_blob_name: str, content_type: str = 'image/png') -> Optional[str]:
    if not bucket:
        logger.error("Firebase Storage bucket not configured. Cannot upload.")
        return None
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(file_bytes, content_type=content_type)
        blob.make_public()
        logger.info(f"Successfully uploaded {destination_blob_name} to {bucket.name}.")
        return blob.public_url
    except Exception as e:
        logger.error(f"Failed to upload {destination_blob_name} to Firebase Storage: {e}", exc_info=True)
        return None

# --- Image Generation Endpoint ---
@app.post("/api/generate-images")
async def generate_multiple_images(request: GenerateRequest):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=503, detail="Replicate API is not configured")
    if not bucket:
        raise HTTPException(status_code=503, detail="Image storage service is not configured")

    try:
        watermark_image = Image.open("watermark.png").convert("RGBA")
        logger.info("Watermark image 'watermark.png' loaded successfully.")
    except FileNotFoundError:
        logger.error("FATAL: 'watermark.png' not found. Cannot process images.")
        raise HTTPException(status_code=500, detail="Server is missing required watermark resource.")

    processed_images: List[Dict[str, str]] = []
    generation_attempts = 3
    success_count = 0
    errors = []

    logger.info(f"Request to generate {generation_attempts} images for prompt: '{request.prompt}'")

    for i in range(generation_attempts):
        logger.info(f"Attempting generation {i+1}/{generation_attempts}...")
        try:
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",
                input={
                    "prompt": request.prompt,
                    "aspect_ratio": request.aspect_ratio,
                    "output_format": "png", "output_quality": 80,
                    "safety_tolerance": 2, "prompt_upsampling": True
                }
            )
            
            temp_image_url = str(output[0]) if isinstance(output, list) and output else str(output)

            if not (temp_image_url and temp_image_url.startswith('https')):
                raise ValueError(f"Invalid temporary URL received: {temp_image_url}")
            
            logger.info(f"Generation {i+1} successful, got temp URL: {temp_image_url}")

            response = requests.get(temp_image_url)
            response.raise_for_status()
            original_image_bytes = response.content

            base_image = Image.open(io.BytesIO(original_image_bytes)).convert("RGBA")
            watermarked_image = base_image.copy()

            padding = 20
            wm_w, wm_h = watermark_image.size
            base_w, base_h = watermarked_image.size
            position = (base_w - wm_w - padding, base_h - wm_h - padding)
            watermarked_image.paste(watermark_image, position, watermark_image)

            with io.BytesIO() as output_bytes:
                watermarked_image.save(output_bytes, format="PNG")
                watermarked_image_bytes = output_bytes.getvalue()

            unique_id = uuid.uuid4()
            original_filename = f"generated/original_{unique_id}.png"
            watermarked_filename = f"generated/watermarked_{unique_id}.png"
            
            original_fb_url = upload_to_firebase_storage(original_image_bytes, original_filename)
            watermarked_fb_url = upload_to_firebase_storage(watermarked_image_bytes, watermarked_filename)

            if original_fb_url and watermarked_fb_url:
                processed_images.append({
                    "originalUrl": original_fb_url,
                    "watermarkedUrl": watermarked_fb_url
                })
                success_count += 1
                logger.info(f"Generation {i+1} fully processed and uploaded to Firebase Storage.")
            else:
                raise IOError("Failed to upload one or both images to Firebase Storage.")

        except (replicate.exceptions.ReplicateError, requests.exceptions.RequestException, ValueError, IOError, Exception) as e:
            error_msg = f"Error during generation {i+1}: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)

    if not processed_images:
        logger.error(f"Failed to generate any valid images for prompt: '{request.prompt}'. Errors: {errors}")
        raise HTTPException(
            status_code=500,
            detail={"message": "Image generation failed.", "errors": errors}
        )

    logger.info(f"Successfully generated and processed {success_count}/{generation_attempts} images.")
    return {"images": processed_images}

# --- Payment Intent Endpoint ---
@app.post("/create-payment-intent")
async def create_payment_intent(data: Annotated[Dict[str, Any], Body(embed=False)]):
    # ... (rest of the code is unchanged)
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment processing is not configured")
    try:
        total_price = data.get('totalPrice')
        if total_price is None: raise HTTPException(status_code=400, detail="Missing 'totalPrice'")
        if not isinstance(total_price, (int, float)): raise HTTPException(status_code=400, detail="'totalPrice' must be a number.")
        if total_price <= 0: raise HTTPException(status_code=400, detail="'totalPrice' must be > 0.")
        amount_cents = calculate_order_amount_cents(total_price)
        payment_intent = stripe.PaymentIntent.create(amount=amount_cents, currency='eur', automatic_payment_methods={'enabled': True})
        return {'clientSecret': payment_intent.client_secret}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Process Order Endpoint ---
@app.post("/process-order")
async def process_order(cart_items: Annotated[List[Dict[str, Any]], Body(embed=False)]):
    # ... (rest of the code is unchanged)
    logger.info("--- ✅ Received /process-order request ---")
    if not cart_items: return {"status": "warning", "message": "Order processed with empty cart."}
    logger.info("🛒 Cart Items Received:")
    for item in cart_items: logger.info(f"  - Item: {item.get('item', 'N/A')}, Price: €{item.get('price', 0.0):.2f}")
    return {"status": "success", "message": "Order received and processed successfully."}

# --- Webhook Endpoint ---
@app.post("/webhook/shipment", status_code=status.HTTP_201_CREATED, tags=["Webhooks"])
async def receive_shipment_data(shipment_data: ShipmentData, request: Request):
    # ... (rest of the code is unchanged)
    if not db: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Firestore service unavailable.")
    try:
        data_to_store = shipment_data.model_dump()
        data_to_store["received_at"] = datetime.datetime.now(datetime.timezone.utc)
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(shipment_data.reference)
        await doc_ref.set(data_to_store)
        return {"status": "success", "message": "Data received.", "doc_id": doc_ref.id}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# --- Optional: Endpoint to retrieve data ---
@app.get("/shipments/{reference_id}", status_code=status.HTTP_200_OK, tags=["Data Retrieval"])
async def get_shipment_data(reference_id: str):
    # ... (rest of the code is unchanged)
    if not db: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Firestore service unavailable.")
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(reference_id)
        doc = await doc_ref.get()
        if not doc.exists: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shipment not found.")
        return doc.to_dict()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    "http://localhost:8081",
    "https://luxxio.netlify.app",
    "https://luxxio.nl"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
