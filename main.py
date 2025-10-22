from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os

app = FastAPI(title="Phishing Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "swathi6016/phishing-detector1")
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

# Pydantic model for request validation
class URLRequest(BaseModel):
    url: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Phishing Detection API",
        "status": "running",
        "model": "DistilBERT via HuggingFace",
        "endpoints": {
            "check": "POST /check",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model": HF_MODEL_ID,
        "hf_token_set": bool(HF_TOKEN)
    }

@app.post("/check")
async def check_url(request: URLRequest):
    """Check if URL is phishing"""
    
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN not configured"
        )
    
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {"inputs": url}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)
            
            if response.status_code == 503:
                raise HTTPException(
                    status_code=503,
                    detail="Model is loading. Please try again in 20 seconds."
                )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"HuggingFace API error: {response.text}"
                )
            
            result = response.json()
            
            # Parse response
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0] if isinstance(result[0], list) else result
                
                phishing_score = 0.0
                legitimate_score = 0.0
                
                for pred in predictions:
                    label = str(pred.get("label", "")).lower()
                    score = float(pred.get("score", 0.0))
                    
                    if "1" in label or "phishing" in label:
                        phishing_score = score
                    elif "0" in label or "legitimate" in label or "legit" in label:
                        legitimate_score = score
                
                is_phishing = phishing_score > legitimate_score
                confidence = max(phishing_score, legitimate_score)
                
                if phishing_score > 0.8:
                    risk_level = "HIGH RISK"
                elif phishing_score > 0.5:
                    risk_level = "MEDIUM RISK"
                else:
                    risk_level = "LOW RISK"
                
                return {
                    "url": url,
                    "is_phishing": is_phishing,
                    "phishing_probability": phishing_score,
                    "legitimate_probability": legitimate_score,
                    "confidence": confidence,
                    "prediction": "PHISHING" if is_phishing else "LEGITIMATE",
                    "risk_level": risk_level
                }
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Unexpected response format from model"
                )
    
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timeout")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Connection error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)