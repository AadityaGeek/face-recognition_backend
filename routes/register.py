from fastapi import APIRouter, UploadFile, Form
from deepface import DeepFace
import qrcode, io, base64
from bson.binary import Binary
import numpy as np
import cv2

import time
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

def resize_frame(frame, max_dim=640):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

@router.post("/register")
async def register_user(
    file: UploadFile,
    name: str = Form(...),
    age: int = Form(...),
    user_id: str = Form(...)
):
    t_start = time.perf_counter()
    print(f"\n--- [REGISTER START] user_id: {user_id} ---")

    t0 = time.perf_counter()
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("  [ERROR] Invalid image provided.")
        return {"success": False, "error": "Invalid image"}

    img = resize_frame(img, max_dim=640)
    print(f"  [1/4] Image Read, Decode & Resize: {(time.perf_counter() - t0)*1000:.1f} ms")

    try:
        t0 = time.perf_counter()
        embedding = DeepFace.represent(
            img,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )[0]["embedding"]
        print(f"  [2/4] DeepFace Feature Extraction: {(time.perf_counter() - t0)*1000:.1f} ms")
    except Exception as e:
        print(f"  [ERROR] Face embedding failed: {str(e)}")
        return {"success": False, "error": f"Face embedding failed: {str(e)}"}

    # Check for duplicates using cosine similarity
    t0 = time.perf_counter()
    for existing in users_collection.find():
        if "embedding" not in existing:
            continue
        sim = cosine_similarity(embedding, existing["embedding"])
        if sim >= DUPLICATE_THRESHOLD:
            print(f"  [WARN] Duplicate face detected! Matches user_id: {existing['user_id']}")
            return {
                "success": False,
                "error": "User already registered",
                "existing_user": {
                    "user_id": existing["user_id"],
                    "name": existing["name"],
                    "age": existing["age"]
                }
            }
    print(f"  [3/4] Duplicate Check in DB: {(time.perf_counter() - t0)*1000:.1f} ms")

    # Build user model
    t0 = time.perf_counter()
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
    print(f"  [4/4] Save to DB & QR Generation: {(time.perf_counter() - t0)*1000:.1f} ms")

    t_total = (time.perf_counter() - t_start) * 1000
    print(f"--- [REGISTER COMPLETE] Total Time: {t_total:.1f} ms ---\n")

    return {"success": True, "qr_code": qr_base64}
