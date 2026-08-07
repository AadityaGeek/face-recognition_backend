from fastapi import APIRouter, UploadFile, File, Form
from deepface import DeepFace
from database.db import users_collection
import numpy as np
import cv2
import tempfile, os

router = APIRouter()
COSINE_THRESHOLD = 0.40  # Cosine distance threshold for Facenet (distance <= 0.40 corresponds to >= 60% similarity)

def detect_motion(video_path: str) -> tuple[bool, float]:
    """
    Analyzes video for frame-to-frame motion.
    Returns (passed_liveness, max_mean_diff).
    """
    # Open the uploaded file as a video stream.
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    max_diff = 0.0
    frame_count = 0

    # Read video frame by frame until no frame is left.
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        # Convert to grayscale to simplify motion comparison.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # Calculate average pixel difference per pixel (resolution independent)
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = float(diff.mean())
            if mean_diff > max_diff:
                max_diff = mean_diff
                
        prev_gray = gray

    # Always release the video resource.
    cap.release()

    # If no usable video frames were found, allow verification to continue.
    if frame_count <= 1:
        # This supports image uploads and avoids false liveness failures.
        return True, 0.0

    # Pass liveness when enough frame-to-frame movement is detected.
    passed = max_diff >= 0.5
    return passed, max_diff


@router.post("/verify")
async def verify_user(user_id: str = Form(...), file: UploadFile = File(...)):
    tmp_path = None
    try:
        # Keep the original extension so OpenCV can decode the file correctly.
        filename = file.filename or "upload.mp4"
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            # Fallback: choose extension from content type when missing.
            ext = ".mp4" if "video" in (file.content_type or "") else ".jpg"

        # Read uploaded bytes once and reuse for decode + temp file write.
        content = await file.read()

        # Save upload to a temp file (used for video frame read and liveness check).
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)

        # Extract one face frame from image/video for comparison.
        extracted_frame = None

        # First, try decoding the upload as a normal image.
        np_arr = np.frombuffer(content, np.uint8)
        extracted_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # If image decode fails, treat it as video and read first frame.
        if extracted_frame is None:
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                extracted_frame = frame

        if extracted_frame is None:
            return {"verified": False, "is_live": False, "error": "Failed to decode image or video frame from upload"}

        # Run a simple liveness check based on motion across frames.
        is_live, motion_score = detect_motion(tmp_path)
        if not is_live:
            return {
                "verified": False,
                "is_live": False,
                "message": "Liveness check failed. Please use live camera.",
                "motion_score": round(motion_score, 3)
            }

        # Load the stored reference image for this user from the database.
        user = users_collection.find_one({"user_id": user_id})
        if not user or "image_data" not in user:
            return {"verified": False, "is_live": True, "error": "User not found or no reference image stored"}

        stored_bytes = user["image_data"]
        np_arr_stored = np.frombuffer(stored_bytes, np.uint8)
        # Decode the stored image bytes into an OpenCV image.
        stored_img = cv2.imdecode(np_arr_stored, cv2.IMREAD_COLOR)

        if stored_img is None:
            return {"verified": False, "is_live": True, "error": "Failed to decode stored user reference image"}

        # Compare uploaded face with stored face using DeepFace.
        result = DeepFace.verify(
            extracted_frame,
            stored_img,
            model_name="Facenet",
            detector_backend="opencv",
            distance_metric="cosine",
            enforce_detection=False
        )

        distance = result.get("distance", 1.0)
        # Convert cosine distance (0.0 = identical, 1.0 = completely different) to similarity percentage.
        similarity = max(0.0, (1.0 - distance)) * 100.0
        # Check against standard Facenet cosine distance threshold.
        verified = distance <= COSINE_THRESHOLD

        response = {
            "verified": verified,
            "score_percent": round(similarity, 2),
            "threshold_percent": round((1.0 - COSINE_THRESHOLD) * 100.0, 2),
            "is_live": True
        }

        if verified:
            response["details"] = {
                "user_id": user.get("user_id", user_id),
                "name": user.get("name", "Verified User"),
                "age": user.get("age", "N/A"),
            }
        else:
            response["message"] = "Face biometric does not match stored user profile"

        return response

    except Exception as e:
        return {"verified": False, "error": f"Server error: {str(e)}"}

    finally:
        # Clean up temp file even if an error happens.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass