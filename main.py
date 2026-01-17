import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from openai import OpenAI

# --- 1. CONFIGURATION ---
class Settings(BaseSettings):
    openai_api_key: str
    app_env: str = "production"
    # Strict timeouts to prevent hanging processes
    openai_timeout: float = 5.0 
    
    class Config:
        env_file = ".env"

# Load settings once
settings = Settings()

# --- 2. LOGGING SETUP ---
# In production, logs usually go to a centralized aggregator (Datadog, ELK).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AIaaS")

# --- 3. SECURITY & RATE LIMITING ---
limiter = Limiter(key_func=get_remote_address)

# Hardened System Prompt (Zero-Trust)
SYSTEM_PROMPT = """
You are the Absurdity Engine. 
Your task: Generate one short, surreal, philosophical, and illogical idea about life.
Constraints:
- Max 2 sentences.
- No advice, no greetings.
- Output ONLY the idea.
"""

# --- 4. LIFECYCLE MANAGEMENT ---
# Efficiently manage the OpenAI client connection
clients = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize Client
    logger.info("Initializing OpenAI Client...")
    clients["openai"] = OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout
    )
    yield
    # Shutdown: Clean up
    logger.info("Shutting down...")
    clients.clear()

# --- 5. APP INITIALIZATION ---
app = FastAPI(
    title="Absurd Idea as a Service",
    version="1.0.0",
    lifespan=lifespan
)

# SECURITY: Add Rate Limit Exception Handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# SECURITY: Trusted Host Middleware
# Prevents HTTP Host Header attacks. In a real domain, replace "*" with "yourdomain.com"
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"] 
)

# SECURITY: CORS Middleware
# Controls which frontend domains can fetch data from this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to ["https://your-frontend.com"] in real prod
    allow_methods=["GET"], # Only allow GET requests
    allow_headers=["*"],
)

# --- 6. DATA MODELS ---
class IdeaResponse(BaseModel):
    idea: str = Field(..., description="The generated absurd idea")
    timestamp: float = Field(default_factory=time.time)

# --- 7. ENDPOINTS ---

@app.get("/health", status_code=200)
async def health_check():
    """Load Balancers use this to check if the app is alive."""
    return {"status": "ok", "environment": settings.app_env}

@app.get("/idea", response_model=IdeaResponse)
@limiter.limit("5/minute") # Rate limit: 5 requests per IP per minute
async def get_absurd_idea(request: Request):
    """
    Returns a random absurd idea. 
    Protected by Rate Limiting and Zero-Input Prompting.
    """
    client: OpenAI = clients.get("openai")
    
    if not client:
        logger.critical("OpenAI Client not initialized.")
        raise HTTPException(status_code=500, detail="Service configuration error.")

    try:
        start_time = time.time()
        
        # The Secure Call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Generate."} # Static input prevents injection
            ],
            temperature=1.3,
            max_tokens=100,
        )
        
        duration = time.time() - start_time
        logger.info(f"Generated idea in {duration:.2f}s for IP: {request.client.host}")
        
        content = response.choices[0].message.content.strip()
        return {"idea": content}

    except Exception as e:
        # LOG the detailed error for the developer
        logger.error(f"OpenAI API Error: {str(e)}", exc_info=True)
        
        # RETURN a generic error to the user (Security: Don't leak stack traces)
        raise HTTPException(
            status_code=503, 
            detail="The absurdity engine is temporarily confused."
        )

# If running directly for debug:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)