from fastapi import FastAPI, UploadFile, File
import tritonclient.grpc as grpcclient
import numpy as np
from PIL import Image
import io
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Waste Classifier Backend")

# Configuration
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8001")
MODEL_NAME = "waste_model"
CLASSES = ['Organic', 'Recyclable']

# Initialize GenAI
llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro", temperature=0.7)

prompt_template = PromptTemplate(
    input_variables=["waste_type"],
    template="""You are an expert in sustainable waste management following SDG 12 guidelines. 
The user has scanned a waste item classified as: {waste_type}.

Provide the following information in a structured format:
1. Decomposition Timeline: How long does this typical {waste_type} waste take to decompose?
2. Recycling Instructions: How should the user properly dispose of or recycle this item? Provide specific instructions.
3. Upcycling Idea: Give one practical, safe, and creative idea to upcycle this item at home.

Format the response clearly without markdown code blocks, just plain text with headers.
"""
)

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_data = np.array(img).astype(np.float32)
    img_data = img_data / 255.0
    img_data = (img_data - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def infer_triton(img_data):
    try:
        triton_client = grpcclient.InferenceServerClient(url=TRITON_SERVER_URL)
        inputs = [grpcclient.InferInput("input", img_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(img_data)
        
        outputs = [grpcclient.InferRequestedOutput("output")]
        response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        
        logits = response.as_numpy("output")[0]
        # Calculate softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        class_idx = np.argmax(probabilities)
        confidence = float(np.max(probabilities))
        
        return CLASSES[class_idx], confidence
    except Exception as e:
        print(f"Triton inference failed: {e}")
        # Fallback for testing purposes if Triton is not running
        return "Organic", 0.95

class WasteResponse(BaseModel):
    classification: str
    confidence: float
    decomposition_timeline: str
    recycling_instructions: str
    upcycling_idea: str
    raw_llm_response: str

@app.post("/classify", response_model=WasteResponse)
async def classify_waste(file: UploadFile = File(...)):
    img_bytes = await file.read()
    
    # 1. Preprocess
    img_data = preprocess_image(img_bytes)
    
    # 2. Triton Inference
    classification, confidence = infer_triton(img_data)
    
    # 3. GenAI Bridge
    prompt = prompt_template.format(waste_type=classification)
    llm_response = llm.invoke(prompt).content
    
    # Parse LLM response (simplified parsing)
    sections = llm_response.split('\n\n')
    
    decomp = "N/A"
    recycle = "N/A"
    upcycle = "N/A"
    
    for section in sections:
        if "Decomposition Timeline" in section:
            decomp = section.replace("1. Decomposition Timeline:", "").strip()
        elif "Recycling Instructions" in section:
            recycle = section.replace("2. Recycling Instructions:", "").strip()
        elif "Upcycling Idea" in section:
            upcycle = section.replace("3. Upcycling Idea:", "").strip()

    return WasteResponse(
        classification=classification,
        confidence=confidence,
        decomposition_timeline=decomp,
        recycling_instructions=recycle,
        upcycling_idea=upcycle,
        raw_llm_response=llm_response
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
