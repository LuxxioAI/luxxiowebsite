from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
from dotenv import load_dotenv

# Finding variables
load_dotenv()

# FastAPI app
app = FastAPI()


class generateRequest(BaseModel):
    prompt: str
    aspect_ratio: str



@app.get("/home")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


@app.get("/generate")
async def generate_image(prompt: str, aspect_ratio: str):

    input = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": "png",
            "output_quality": 80,
            "safety_tolerance": 2,
            "prompt_upsampling": True
    }
    try:
        # Send POST request (replace with actual request method if needed)
        output = replicate.run(
            "black-forest-labs/flux-1.1-pro",
            input=input,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")

    # Handle response (replace with logic to process actual response data)
    return {"img": output}
