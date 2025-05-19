# main.py
import os
import stripe # type: ignore
from fastapi import FastAPI, HTTPException, Body, Request, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field # Ensure BaseModel and Field are imported
import logging
import uvicorn
import replicate # Assuming this is used elsewhere
from google.cloud import firestore # Assuming this is used elsewhere
import datetime # Assuming this is used elsewhere

# --- Load Environment Variables ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Firestore Configuration (example, adapt if needed)
try:
    db = firestore.AsyncClient()
    logger.info("Firestore client initialized successfully.")
    FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION_NAME", "shipments")
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {e}")
    db = None

STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')
if not STRIPE_SECRET_KEY:
    logger.critical("🔴 FATAL ERROR: Stripe Secret Key not found in .env file.")
    exit(1)
stripe.api_key = STRIPE_SECRET_KEY
logger.info("✅ Stripe API Key Loaded.")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN") # Keep if used

# --- FastAPI App ---
app = FastAPI()

# --- Pydantic Models ---
class GenerateRequest(BaseModel): # Keep if used
    prompt: str
    aspect_ratio: str

class ShipmentData(BaseModel): # Keep if used
    reference: str = Field(..., example="test001")
    orderid: str = Field(default="", example="")
    shipment_company: str = Field(..., example="DPD")
    tracking_code: str = Field(..., example="05132088424538")
    tracking_url: str = Field(..., example="https://www.dpdgroup.com/nl/mydpd/my-parcels/search?lang=nl")

class CreatePaymentIntentRequest(BaseModel):
    totalPrice: float

class VerifyPaymentRequest(BaseModel): # New model for the verification request
    payment_intent_id: str = Field(..., example="pi_xxxxxxxxxxxxxxxxx")


# --- Helper Function ---
def calculate_order_amount_cents(total_price_euros: float) -> int:
    if total_price_euros <= 0:
        raise ValueError("Total price must be positive.")
    return int(round(total_price_euros * 100))

# --- Endpoints ---

@app.get("/")
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {"greeting": "Hello, World!", "message": "Welcome to the API!"}

# Image Generation Endpoint (Keep as is from your provided code)
@app.post("/api/generate-images")
async def generate_multiple_images(request: GenerateRequest):
    # ... (Your existing generate_multiple_images logic)
    image_urls: List[str] = []
    generation_attempts = 3
    success_count = 0
    errors = []

    logger.info(f"Received request to generate {generation_attempts} images for prompt: '{request.prompt}' with aspect ratio: {request.aspect_ratio}")

    for i in range(generation_attempts):
        logger.info(f"Attempting generation {i+1}/{generation_attempts}...")
        try:
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
            image_url = None
            result_item = None
            if isinstance(output, list) and len(output) > 0:
                result_item = output[0]
            elif output:
                result_item = output
            
            if result_item:
                try:
                    image_url = str(result_item)
                    if isinstance(image_url, str) and image_url.startswith(('http://', 'https://')):
                        image_urls.append(image_url)
                        success_count += 1
                    else:
                        errors.append(f"Generation {i+1}: Converted result is not a valid URL string. Value: {image_url}")
                except Exception as convert_e:
                    errors.append(f"Generation {i+1}: Error converting result item to string: {convert_e}")
            else:
                 errors.append(f"Generation {i+1}: No valid result item found in output.")
        except replicate.exceptions.ReplicateError as e:
            errors.append(f"Replicate API error during generation {i+1}: {e}")
        except Exception as e:
            errors.append(f"Unexpected error during generation {i+1}: {e}")

    if not image_urls:
        raise HTTPException(
            status_code=500,
            detail={"message": f"Image generation failed after {generation_attempts} attempts.", "errors": errors}
        )
    return {"images": image_urls}


@app.post("/create-payment-intent")
async def create_payment_intent(data: CreatePaymentIntentRequest): # Use Pydantic model
    try:
        total_price = data.totalPrice # Access directly due to Pydantic model
        logger.info(f"ℹ️ Received /create-payment-intent request with totalPrice: {total_price}")

        if not isinstance(total_price, (int, float)):
            logger.error("🔴 Validation Error: 'totalPrice' is not a number.")
            raise HTTPException(status_code=400, detail="'totalPrice' must be a number.")
        if total_price <= 0:
             logger.error("🔴 Validation Error: 'totalPrice' must be positive.")
             raise HTTPException(status_code=400, detail="'totalPrice' must be greater than zero.")

        amount_cents = calculate_order_amount_cents(total_price)
        logger.info(f"💰 Calculated amount in cents: {amount_cents}")

        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency='eur',
            automatic_payment_methods={'enabled': True},
        )
        logger.info(f"✅ Successfully created PaymentIntent: {payment_intent.id}")
        return {'clientSecret': payment_intent.client_secret}

    except ValueError as ve:
        logger.error(f"🔴 Value Error during amount calculation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except stripe.error.StripeError as e:
        logger.error(f"🔴 Stripe API Error: {e.user_message or str(e)}")
        raise HTTPException(status_code=e.http_status or 400, detail=e.user_message or "A payment processing error occurred.")
    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.error(f"💥 Unexpected Server Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- NEW ENDPOINT to verify payment intent ---
@app.post("/verify-payment-intent")
async def verify_payment_intent_endpoint(request_data: VerifyPaymentRequest):
    try:
        payment_intent_id = request_data.payment_intent_id
        if not payment_intent_id or not payment_intent_id.startswith("pi_"):
            logger.warning(f"Invalid Payment Intent ID format received: {payment_intent_id}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid Payment Intent ID format.")

        logger.info(f"Verifying PaymentIntent from backend: {payment_intent_id}")
        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        logger.info(f"PaymentIntent {payment_intent_id} status: {intent.status}")
        
        # Return relevant fields from the PaymentIntent
        return {
            "id": intent.id,
            "status": intent.status,
            "amount": intent.amount,
            "currency": intent.currency,
            # Add any other fields you might need on the frontend
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe API Error verifying PI {request_data.payment_intent_id}: {e.user_message or str(e)}")
        raise HTTPException(status_code=e.http_status or status.HTTP_400_BAD_REQUEST, detail=e.user_message or "Error verifying payment with Stripe.")
    except HTTPException as http_exc: # Re-raise existing HTTPExceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error verifying PI {request_data.payment_intent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while verifying payment.")


@app.post("/process-order") # Keep as is
async def process_order(
    cart_items: Annotated[List[Dict[str, Any]], Body(embed=False)]
):
    # ... (Your existing process_order logic)
    logger.info("\n--- ✅ Received /process-order request ---")
    if not cart_items:
        logger.warning("⚠️ Received empty cart for processing.")
        return {"status": "warning", "message": "Order processed with empty cart."}
    # ... rest of your logic
    return {"status": "success", "message": "Order received and processed successfully."}


@app.post("/webhook/shipment", status_code=status.HTTP_201_CREATED, tags=["Webhooks"]) # Keep as is
async def receive_shipment_data(shipment_data: ShipmentData, request: Request):
    # ... (Your existing webhook logic)
    if not db:
        raise HTTPException(status_code=503, detail="Firestore not available")
    data_to_store = shipment_data.model_dump()
    data_to_store["received_at"] = datetime.datetime.now(datetime.timezone.utc)
    doc_ref = db.collection(FIRESTORE_COLLECTION).document(shipment_data.reference)
    await doc_ref.set(data_to_store)
    return {"status": "success", "message": "Shipment data received"}


@app.get("/shipments/{reference_id}", status_code=status.HTTP_200_OK, tags=["Data Retrieval"]) # Keep as is
async def get_shipment_data(reference_id: str):
    # ... (Your existing get_shipment_data logic)
    if not db:
        raise HTTPException(status_code=503, detail="Firestore not available")
    doc_ref = db.collection(FIRESTORE_COLLECTION).document(reference_id)
    doc_snapshot = await doc_ref.get()
    if not doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="Shipment not found")
    return doc_snapshot.to_dict()


# --- CORS Middleware (Keep as before) ---
origins = [
    "http://localhost:3000",
    "http://localhost:8081",
    "https://luxxio.netlify.app",
    "https://luxxio.nl"
    # Add your Expo Go URL if needed, e.g., "exp://<your-ip>:<port>"
    # Add your production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- How to run (Keep as before) ---
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
