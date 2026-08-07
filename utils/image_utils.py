import cv2
import numpy as np

def resize_frame(frame, max_dim=640):
    """
    Resizes image frame so max(width, height) <= max_dim.
    Preserves aspect ratio.
    """
    if frame is None:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame

def cosine_similarity(vec1, vec2) -> float:
    """
    Computes cosine similarity between two feature vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
