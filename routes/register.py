from fastapi import APIRouter, UploadFile, Form
from deepface import DeepFace
import qrcode, io, base64
from bson.binary import Binary
import numpy as np
import cv2

from database.db import users_collection
from models.user import User

router = APIRouter()

# threshold for duplicate detection (cosine similarity)
DUPLICATE_THRESHOLD = 0.4  # 40%


@router.get("/check-user-id")
async def check_user_id(user_id: str):
    user = users_collection.find_one({"user_id": user_id}, {"_id": 1})
    return {"exists": user is not None, "user_id": user_id}

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@router.post("/register")
async def register_user(
    file: UploadFile,
    name: str = Form(...),
    age: int = Form(...),
    user_id: str = Form(...)
):
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "Invalid image"}

    try:
        embedding = DeepFace.represent(
            img,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]
    except Exception as e:
        return {"success": False, "error": f"Face embedding failed: {str(e)}"}

    # Check for duplicates using cosine similarity
    for existing in users_collection.find():
        if "embedding" not in existing:
            continue
        sim = cosine_similarity(embedding, existing["embedding"])
        if sim >= DUPLICATE_THRESHOLD:
            return {
                "success": False,
                "error": "User already registered",
                "existing_user": {
                    "user_id": existing["user_id"],
                    "name": existing["name"],
                    "age": existing["age"]
                }
            }

    # Build user model
    user = User(
        user_id=user_id,
        name=name,
        age=age,
        embedding=embedding,
        image_path=""
    )

    users_collection.insert_one({
        **user.dict(),
        "image_data": Binary(file_bytes)
    })

    # Generate QR code
    qr = qrcode.make(user_id)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    qr_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"success": True, "qr_code": qr_base64}
