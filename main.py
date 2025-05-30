# main.py
import os
import base64
import json
import tempfile
import stripe # type: ignore # Ignore type error if stripe package typing is missing
from fastapi import FastAPI, HTTPException, Body, Request, status
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, Optional
from google.cloud import firestore
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

# --- Firestore Credentials Setup ---
def setup_firestore_credentials():
    """Setup Firestore credentials from Base64 encoded environment variable"""
    base64_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    
    if base64_key:
        try:
            logger.info("Setting up Firestore credentials from Base64 environment variable...")
            
            # Clean the Base64 string (remove any whitespace/newlines)
            base64_key = base64_key.strip().replace('\n', '').replace('\r', '').replace(' ', '')
            logger.info(f"Base64 key length: {len(base64_key)} characters")
            
            # Decode the Base64 string
            try:
                key_data = base64.b64decode(base64_key).decode('utf-8')
            except Exception as decode_error:
                logger.error(f"Base64 decode error: {decode_error}")
                logger.error(f"First 100 chars of Base64: {base64_key[:100]}...")
                raise
            
            # Parse JSON
            try:
                key_json = json.loads(key_data)
            except Exception as json_error:
                logger.error(f"JSON parse error: {json_error}")
                logger.error(f"First 200 chars of decoded data: {key_data[:200]}...")
                raise
            
            # Validate required fields
            required_fields = ['type', 'project_id', 'private_key', 'client_email']
            missing_fields = [field for field in required_fields if field not in key_json]
            if missing_fields:
                logger.error(f"Missing required fields in service account JSON: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            # Log some info (without sensitive data)
            logger.info(f"Service account email: {key_json.get('client_email')}")
            logger.info(f"Project ID: {key_json.get('project_id')}")
            
            # Validate private key format
            private_key = key_json.get('private_key', '')
            if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
                logger.error("Private key does not start with proper header")
                logger.error(f"Private key starts with: {private_key[:50]}...")
                raise ValueError("Invalid private key format")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(key_json, temp_file, indent=2)
                temp_file_path = temp_file.name
            
            # Set the environment variable to point to the temp file
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_file_path
            logger.info(f"Firestore credentials file created at: {temp_file_path}")
            
            # Test the credentials by trying to create a client
            try:
                test_client = firestore.Client()
                logger.info("✅ Firestore credentials validated successfully")
                test_client.close()
            except Exception as test_error:
                logger.error(f"❌ Firestore credentials validation failed: {test_error}")
                raise
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"Failed to setup Firestore credentials from Base64: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            return None
    else:
        logger.info("No Base64 credentials found, checking for existing GOOGLE_APPLICATION_CREDENTIALS...")
        # Check if regular GOOGLE_APPLICATION_CREDENTIALS is set (for local development)
        if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.info("Using existing GOOGLE_APPLICATION_CREDENTIALS file")
            return os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        else:
            logger.warning("No Firestore credentials found in environment variables")
            return None

# Firestore Configuration
try:
    # Setup credentials if Base64 is provided
    credentials_path = setup_firestore_credentials()
    
    if credentials_path:
        db = firestore.AsyncClient()
        logger.info("Firestore client initialized successfully.")
        # Use environment variable for collection name, default to 'shipments'
        FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION_NAME", "shipments")
        logger.info(f"Using Firestore collection: {FIRESTORE_COLLECTION}")
    else:
        logger.error("No valid Firestore credentials found")
        db = None
        FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION_NAME", "shipments")
        
except Exception as e:
    logger.error(f"Failed to initialize Firestore client: {e}")
    # If Firestore cannot be initialized, the app shouldn't start properly.
    # Raising an exception might be appropriate in a real scenario,
    # but for simplicity, we'll let it proceed and endpoints will fail.
    db = None
    FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION_NAME", "shipments")

# --- Stripe Configuration ---
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY')

if not STRIPE_SECRET_KEY:
    logger.error("🔴 FATAL ERROR: Stripe Secret Key not found in .env file.")
    # In production, you might want to exit here
    # exit(1) # Exit if key is missing
else:
    # Initialize Stripe
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe API Key Loaded.")

# --- Replicate Configuration ---
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN environment variable not set!")
else:
    logger.info("Replicate API token found.")

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
    reference: str = Field(..., example="test001") # '...' means required
    orderid: str = Field(default="", example="") # Allow empty string, provide default
    shipment_company: str = Field(..., example="DPD")
    tracking_code: str = Field(..., example="05132088424538")
    tracking_url: str = Field(..., example="https://www.dpdgroup.com/nl/mydpd/my-parcels/search?lang=nl")

# --- Helper Function ---
def calculate_order_amount_cents(total_price_euros: float) -> int:
    """Converts a price in Euros (float) to cents (integer)."""
    # Ensure price is positive before calculation
    if total_price_euros <= 0:
        raise ValueError("Total price must be positive.")
    return int(round(total_price_euros * 100)) # Use round for better precision

# --- Root Endpoint ---
@app.get("/")
async def root():
    logger.info("Root endpoint '/' accessed.")
    return {
        "greeting": "Hello, World!", 
        "message": "Welcome to the Image Generation API!",
        "firestore_status": "connected" if db else "disconnected",
        "stripe_status": "configured" if STRIPE_SECRET_KEY else "not configured",
        "replicate_status": "configured" if REPLICATE_API_TOKEN else "not configured"
    }

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "services": {
            "firestore": "connected" if db else "disconnected",
            "stripe": "configured" if STRIPE_SECRET_KEY else "not configured",
            "replicate": "configured" if REPLICATE_API_TOKEN else "not configured"
        }
    }

# --- Debug Endpoint (Remove in production) ---
@app.get("/debug/firestore")
async def debug_firestore():
    """Debug endpoint to check Firestore configuration"""
    debug_info = {
        "firestore_client_exists": db is not None,
        "collection_name": FIRESTORE_COLLECTION,
        "google_application_credentials_set": bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS')),
        "google_application_credentials_base64_set": bool(os.getenv('GOOGLE_APPLICATION_CREDENTIALS_BASE64')),
    }
    
    # Try to get the length of base64 string without exposing it
    base64_key = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
    if base64_key:
        debug_info["base64_key_length"] = len(base64_key.strip())
        debug_info["base64_starts_with"] = base64_key[:20] + "..."
    
    # Test Firestore connection
    if db:
        try:
            # Fix: Use list comprehension with async generator instead of await
            collections = [col.id async for col in db.collections()]
            debug_info["firestore_test"] = "success"
            debug_info["available_collections"] = collections
        except Exception as e:
            debug_info["firestore_test"] = f"failed: {str(e)}"
    
    return debug_info
    
# --- Image Generation Endpoint ---
@app.post("/api/generate-images")
async def generate_multiple_images(request: GenerateRequest):
    """
    Generates 3 images using the Replicate API based on the provided prompt
    and aspect ratio. Returns an array of image URLs.
    Handles FileOutput object from Replicate.
    """
    if not REPLICATE_API_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Replicate API is not configured"
        )
    
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

# --- Payment Intent Endpoint ---
@app.post("/create-payment-intent")
async def create_payment_intent(
    # Use Annotated for clear Body parsing with type hints
    data: Annotated[Dict[str, Any], Body(
        embed=False, # Don't require data to be nested under a key
        examples=[{"totalPrice": 49.99}] # Example for documentation
    )]
):
    """
    Creates a Stripe PaymentIntent based on the total price provided.
    Returns the clientSecret needed by the frontend Stripe Elements.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=503,
            detail="Payment processing is not configured"
        )
    
    try:
        total_price = data.get('totalPrice')
        logger.info(f"ℹ️ Received /create-payment-intent request with totalPrice: {total_price}")

        # --- Input Validation ---
        if total_price is None:
            logger.warning("🔴 Validation Error: 'totalPrice' missing.")
            raise HTTPException(status_code=400, detail="Missing 'totalPrice' in request body.")
        if not isinstance(total_price, (int, float)):
            logger.warning("🔴 Validation Error: 'totalPrice' is not a number.")
            raise HTTPException(status_code=400, detail="'totalPrice' must be a number.")
        if total_price <= 0:
             logger.warning("🔴 Validation Error: 'totalPrice' must be positive.")
             raise HTTPException(status_code=400, detail="'totalPrice' must be greater than zero.")

        # --- Calculate Amount ---
        amount_cents = calculate_order_amount_cents(total_price)
        logger.info(f"💰 Calculated amount in cents: {amount_cents}")

        # --- Create PaymentIntent ---
        payment_intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency='eur',  # Adjust currency if needed (e.g., 'usd')
            # Enable automatic collection of payment methods shown by Stripe Elements
            automatic_payment_methods={
                'enabled': True,
            },
            # You can add metadata here if needed (e.g., order ID, user ID)
            # metadata={'order_id': 'some_order_id', 'user_id': 'user_123'}
        )
        logger.info(f"✅ Successfully created PaymentIntent: {payment_intent.id}")

        # --- Return Client Secret ---
        return {'clientSecret': payment_intent.client_secret}

    except ValueError as ve: # Catch specific calculation errors
        logger.error(f"🔴 Value Error during amount calculation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except stripe.error.StripeError as e:
        # Handle specific Stripe API errors
        logger.error(f"🔴 Stripe API Error: {e.user_message}") # Log user-friendly message
        raise HTTPException(status_code=400, detail=e.user_message or "A payment processing error occurred.")
    except HTTPException as http_exc:
         # Re-raise validation exceptions directly
         raise http_exc
    except Exception as e:
        # Handle unexpected server errors
        logger.error(f"💥 Unexpected Server Error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# --- Process Order Endpoint ---
@app.post("/process-order")
async def process_order(
    # Expects a list of dictionaries directly in the body
    cart_items: Annotated[List[Dict[str, Any]], Body(
         embed=False, # Don't require list to be nested under a key
         examples=[[{"id": "prod_1", "name": "Canvas Print", "price": 25.50}]] # Example
    )]
):
    """
    Placeholder endpoint called after successful payment confirmation.
    Receives the cart items. In a real app, saves order to database,
    triggers fulfillment, sends emails, etc.
    """
    logger.info("--- ✅ Received /process-order request ---")
    if not cart_items:
        logger.warning("⚠️ Received empty cart for processing.")
        # Decide if this is an error or acceptable
        return {"status": "warning", "message": "Order processed with empty cart."}

    logger.info("🛒 Cart Items Received:")
    for item in cart_items:
        # Log some details - adapt based on PushedProduct structure
        logger.info(f"  - ID: {item.get('id', 'N/A')}, "
              f"Item: {item.get('item', 'N/A')}, "
              f"Dimensions: {item.get('dimensions', 'N/A')}, "
              f"Price: €{item.get('price', 0.0):.2f}")

    # --- Placeholder for Real Actions ---
    # 1. Verify Payment Status again (optional, more secure)
    #    - You could potentially receive the paymentIntentId here too
    #    - Retrieve PI from Stripe: stripe.PaymentIntent.retrieve(payment_intent_id)
    #    - Check if status is 'succeeded' before saving
    # 2. Save order details (cart_items, user info if available) to your database.
    # 3. Trigger fulfillment process (e.g., notify warehouse).
    # 4. Send confirmation email to the customer.
    # --- End Placeholder ---

    logger.info("--- ✅ Order processing simulation complete ---")
    return {"status": "success", "message": "Order received and processed successfully."}

# --- Webhook Endpoint ---
@app.post("/webhook/shipment", status_code=status.HTTP_201_CREATED, tags=["Webhooks"])
async def receive_shipment_data(shipment_data: ShipmentData, request: Request):
    """
    Receives shipment data via POST request and stores it in Firestore.
    Uses the 'reference' field as the Firestore document ID.
    If a document with the same 'reference' already exists, it will be overwritten.
    """
    if not db:
        logger.error("Firestore client is not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore service is not configured or unavailable.",
        )

    try:
        # Log received data
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"Received shipment data from {client_host} for reference: {shipment_data.reference}")
        logger.debug(f"Data: {shipment_data.model_dump_json()}") # Use model_dump_json for Pydantic v2+

        # Prepare data for Firestore (include a timestamp)
        data_to_store = shipment_data.model_dump() # Use model_dump for Pydantic v2+
        data_to_store["received_at"] = datetime.datetime.now(datetime.timezone.utc)

        # Get a reference to the document using 'reference' as the ID
        # This allows updates if the same reference is sent again.
        # If you want a new document every time, use db.collection(...).add() instead.
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(shipment_data.reference)

        # Set the data in Firestore (creates or overwrites)
        await doc_ref.set(data_to_store)

        logger.info(f"Successfully stored/updated data for reference: {shipment_data.reference} in collection '{FIRESTORE_COLLECTION}'")

        return {
            "status": "success",
            "message": "Shipment data received and stored successfully.",
            "firestore_collection": FIRESTORE_COLLECTION,
            "firestore_doc_id": doc_ref.id, # doc_ref.id will be shipment_data.reference
        }

    except Exception as e:
        logger.exception(f"Error processing shipment data for reference {shipment_data.reference}: {e}") # Use logger.exception to include stack trace
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the data: {str(e)}",
        )

# --- Optional: Endpoint to retrieve data (for testing/verification) ---
@app.get("/shipments/{reference_id}", status_code=status.HTTP_200_OK, tags=["Data Retrieval"])
async def get_shipment_data(reference_id: str):
    """Retrieves shipment data from Firestore by its reference ID."""
    if not db:
        logger.error("Firestore client is not available.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Firestore service is not configured or unavailable.",
        )

    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION).document(reference_id)
        doc_snapshot = await doc_ref.get()

        if not doc_snapshot.exists:
            logger.warning(f"Document with reference ID '{reference_id}' not found in collection '{FIRESTORE_COLLECTION}'.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Shipment with reference '{reference_id}' not found.",
            )

        logger.info(f"Retrieved data for reference ID: {reference_id}")
        return doc_snapshot.to_dict()

    except HTTPException:
        # Re-raise HTTPException (like 404 Not Found) directly
        raise
    except Exception as e:
        logger.exception(f"Error retrieving shipment data for reference {reference_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving the data: {str(e)}",
        )

# --- CORS Middleware ---
origins = [
    "http://localhost:3000",
    "http://localhost:8081", # For Expo Go?
    # Add your frontend domains if needed
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

# --- How to run the application (using uvicorn) ---
# In your terminal, run:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
#
# --reload : Automatically restarts the server when code changes (for development)
# --host 0.0.0.0 : Makes the server accessible on your network
# --port 8000 : Specifies the port number

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
