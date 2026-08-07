import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import register, verify

app = FastAPI()

# Dynamic & default CORS origins
allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",")]
else:
    origins = [
        "https://face-recognition-app-self.vercel.app",
        "http://localhost:5173",
        "*"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Backend is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(register.router)
app.include_router(verify.router)

@app.on_event("startup")
async def warmup_deepface_model():
    try:
        import numpy as np
        from deepface import DeepFace
        print("Warming up DeepFace Facenet model on server startup...")
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.represent(
            dummy_img,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        print("DeepFace Facenet model warmed up successfully!")
    except Exception as e:
        print(f"Model warmup notice: {e}")

