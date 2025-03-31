from fastapi import FastAPI, HTTPException
import replicate
from dotenv import find_dotenv, load_dotenv

# Finding variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# FastAPI app
app_fastapi = FastAPI()


@app_fastapi.get("/")
async def root():
    return {"greeting": "Hello, World!", "message": "Welcome to FastAPI!"}


@app_fastapi.post("/api/generate-image")
async def generate_image(prompt: str, aspect_ratio: str):
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

    try:

        # Send POST request (replace with actual request method if needed)
        output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input=input,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {e}")

    # Handle response (replace with logic to process actual response data)
    return {"img": output}


if __name__ == "__main__":
    # FastAPI on port 8000 (modify as needed)
    import uvicorn

    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)
