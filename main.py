import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import logging # Add logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (primarily for local development)
load_dotenv()

# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")

# --- Robust AI Initialization ---
model = None # Initialize model as None
if not api_key:
    logger.warning("GOOGLE_API_KEY not found in environment variables. AI features will be disabled.")
else:
    try:
        genai.configure(api_key=api_key)
        # Consider making the model name an environment variable too for flexibility
        model = genai.GenerativeModel('gemini-2.0-flash') # Use the correct model name
        logger.info("Google AI configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Google AI: {e}. AI features will be disabled.")
        model = None # Ensure model is None if configuration fails

# Define input and output models
class PlotInput(BaseModel):
    plot: str

class TreatmentOutput(BaseModel):
    treatment: str

# Keep generate_treatment function separate, but it will use the global model
async def generate_treatment(plot_text: str) -> str:
    if not model:
        logger.error("Attempted to generate treatment, but AI model is not available.")
        raise HTTPException(status_code=503, detail="AI Model not available") # 503 Service Unavailable

    if not plot_text:
         logger.warning("Generate treatment called with empty plot text.")
         raise HTTPException(status_code=400, detail="Plot cannot be empty")

    prompt = f"""Expand the following story plot into a detailed story in just 250 words. Focus on outlining key scenes, character arcs, and potential turning points.

Plot:
{plot_text}

Treatment:"""
    try:
        logger.info("Generating content with Google AI...")
        response = await model.generate_content_async(prompt)

        if not response or not response.text:
            logger.error("AI returned empty response.")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate story treatment. The model returned an empty response."
            )
        logger.info("Successfully generated treatment.")
        return response.text

    except Exception as e:
        logger.error(f"Error during AI content generation: {e}")
        # Propagate a generic error to the user, but log the specific details
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate story treatment due to an internal error."
        )

app = FastAPI(title="Story Maker API") # Updated title

# --- Update CORS Origins ---
# Add your future GitHub Pages URL (replace placeholder)
# Add Render's preview deployment wildcard
origins = [
    "http://localhost:5173",  # Vue dev server
    "https://*.onrender.com", # Render preview deployments wildcard
    "https://mahadevan.github.io", # Your live GitHub Pages URL (origin only)
    "https://mahadevan.github.io/story-maker-frontend", # Your live GitHub Pages URL (full path, might be needed depending on requests)
    # Add your production frontend URL here later
]

logger.info(f"Configuring CORS for origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/generate-treatment", response_model=TreatmentOutput)
async def create_treatment(plot_input: PlotInput):
    # The endpoint now relies on the globally initialized (or None) model via generate_treatment
    logger.info(f"Received request to generate treatment for plot: '{plot_input.plot[:50]}...'") # Log start of request
    treatment = await generate_treatment(plot_input.plot)
    return TreatmentOutput(treatment=treatment) 