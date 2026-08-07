from fastapi import APIRouter, UploadFile, File, Form
from deepface import DeepFace
from database.db import users_collection
import numpy as np
import cv2
import tempfile, os

router = APIRouter()
COSINE_THRESHOLD = 0.40  # Cosine distance threshold for Facenet (distance <= 0.40 corresponds to >= 60% similarity)

def resize_frame(frame, max_dim=640):
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def detect_motion(video_path: str) -> tuple[bool, float]:
    """
    Analyzes video for frame-to-frame motion by sub-sampling max 15 evenly-spaced frames.
    Returns (passed_liveness, max_mean_diff).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return True, 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 1:
        cap.release()
        return True, 0.0

    max_samples = 15
    step = max(1, total_frames // max_samples)

    prev_gray = None
    max_diff = 0.0
    sampled_count = 0

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Fast downscale to 320px for high-speed motion difference calculation
        h, w = frame.shape[:2]
        if max(h, w) > 320:
            scale = 320.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = float(diff.mean())
            if mean_diff > max_diff:
                max_diff = mean_diff

        prev_gray = gray
        sampled_count += 1
        if sampled_count >= max_samples:
            break

    cap.release()
    passed = max_diff >= 0.5
    return passed, max_diff


import time

@router.post("/verify")
async def verify_user(user_id: str = Form(...), file: UploadFile = File(...)):
    t_start = time.perf_counter()
    print(f"\n--- [VERIFY START] user_id: {user_id} ---")
    tmp_path = None
    try:
        # Keep the original extension so OpenCV can decode the file correctly.
        filename = file.filename or "upload.mp4"
        ext = os.path.splitext(filename)[1].lower()
        if not ext:
            # Fallback: choose extension from content type when missing.
            ext = ".mp4" if "video" in (file.content_type or "") else ".jpg"

        t0 = time.perf_counter()
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

        # If image decode fails, treat it as video and read the Middle Frame (50% midpoint)
        if extracted_frame is None:
            cap = cv2.VideoCapture(tmp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                mid_idx = max(0, total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    extracted_frame = frame
            
            # Fallback if midpoint seek fails
            if extracted_frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    extracted_frame = frame
            cap.release()

        if extracted_frame is None:
            print("  [ERROR] Failed to decode image or video frame from upload")
            return {"verified": False, "is_live": False, "error": "Failed to decode image or video frame from upload"}

        # Resize image for fast OpenCV face detection & feature extraction
        extracted_frame = resize_frame(extracted_frame, max_dim=640)
        print(f"  [1/4] File Read, Middle Frame Select & Resize: {(time.perf_counter() - t0)*1000:.1f} ms")

        # Run a simple liveness check based on motion across frames.
        t0 = time.perf_counter()
        is_live, motion_score = detect_motion(tmp_path)
        print(f"  [2/4] Liveness Motion Check: {(time.perf_counter() - t0)*1000:.1f} ms (Passed: {is_live}, Score: {motion_score:.3f})")
        if not is_live:
            print(f"  [WARN] Liveness check failed for user_id: {user_id}")
            return {
                "verified": False,
                "is_live": False,
                "message": "Liveness check failed. Please use live camera.",
                "motion_score": round(motion_score, 3)
            }

        # Load user record from MongoDB with field projection (skips fetching heavy image binary)
        t0 = time.perf_counter()
        user = users_collection.find_one(
            {"user_id": user_id},
            {"embedding": 1, "name": 1, "age": 1, "user_id": 1}
        )
        print(f"  [3/4] MongoDB Vector Lookup: {(time.perf_counter() - t0)*1000:.1f} ms")
        if not user:
            print(f"  [ERROR] User not found: {user_id}")
            return {"verified": False, "is_live": True, "error": "User not found"}

        if "embedding" not in user:
            print("  [ERROR] User profile missing face biometric data")
            return {"verified": False, "is_live": True, "error": "User profile missing face biometric data"}

        # Extract embedding from uploaded face frame
        t0 = time.perf_counter()
        rep = DeepFace.represent(
            extracted_frame,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        if not rep or len(rep) == 0:
            print("  [ERROR] Could not extract face biometric from upload")
            return {"verified": False, "is_live": True, "error": "Could not extract face biometric from upload"}

        incoming_vec = np.array(rep[0]["embedding"])
        stored_vec = np.array(user["embedding"])

        # Compute Cosine Distance directly using pre-stored vector
        norm1 = np.linalg.norm(incoming_vec)
        norm2 = np.linalg.norm(stored_vec)
        if norm1 == 0 or norm2 == 0:
            distance = 1.0
        else:
            cos_sim = float(np.dot(incoming_vec, stored_vec) / (norm1 * norm2))
            distance = max(0.0, 1.0 - cos_sim)

        # Convert cosine distance (0.0 = identical, 1.0 = completely different) to similarity percentage.
        similarity = max(0.0, (1.0 - distance)) * 100.0
        # Check against standard Facenet cosine distance threshold.
        verified = distance <= COSINE_THRESHOLD
        print(f"  [4/4] DeepFace Feature Extraction & Math: {(time.perf_counter() - t0)*1000:.1f} ms (Dist: {distance:.4f}, Match: {verified})")

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

        t_total = (time.perf_counter() - t_start) * 1000
        print(f"--- [VERIFY COMPLETE] Verified: {verified} | Total Time: {t_total:.1f} ms ---\n")

        return response

    except Exception as e:
        print(f"  [EXCEPT] Server error: {str(e)}")
        return {"verified": False, "error": f"Server error: {str(e)}"}

    finally:
        # Clean up temp file even if an error happens.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass