import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key and check if it exists
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    error_message = "ERROR: GOOGLE_API_KEY not found in environment variables. Make sure it's set in the .env file."
    print(error_message)
    raise RuntimeError(error_message)

# Configure Gemini
genai.configure(api_key=api_key)

# Define input and output models
class PlotInput(BaseModel):
    plot: str

class TreatmentOutput(BaseModel):
    treatment: str

async def generate_treatment(plot_text: str) -> str:
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Define the prompt
        prompt = f"""Expand the following story plot into a detailed story in just 250 words. Focus on outlining key scenes, character arcs, and potential turning points.
        
Plot:
{plot_text}

Treatment:"""

        # Generate content
        response = await model.generate_content_async(prompt)
        
        # Check for successful generation
        if not response or not response.text:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate story treatment. The model returned an empty response."
            )
        
        return response.text

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate story treatment: {str(e)}"
        )

app = FastAPI(title="Plot Maker API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vue.js dev server default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "API is running"}

@app.post("/generate-treatment", response_model=TreatmentOutput)
async def create_treatment(plot_input: PlotInput):
    treatment = await generate_treatment(plot_input.plot)
    return TreatmentOutput(treatment=treatment) 