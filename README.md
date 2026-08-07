# Face Recognition & Verification Backend

FastAPI asynchronous backend service for AI-powered face registration, real-time motion-based liveness verification, and biometric face matching using DeepFace (Facenet) and MongoDB.

---

## 🛠️ Technology Stack

* **Framework**: FastAPI (Python 3.11)
* **Face Recognition & Biometrics**: DeepFace (`Facenet` model, `opencv` detector backend)
* **Computer Vision**: OpenCV (`opencv-python-headless`)
* **Database**: MongoDB (via PyMongo & BSON Binary storage)
* **QR Code Generation**: `qrcode` library & `Pillow`
* **Deployment**: Docker & Railway

---

## 📁 Directory Structure

```text
backend/
├── main.py              # Application entrypoint, CORS configuration & health check routes
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker container production configuration
├── railway.json         # Railway deployment manifest
├── database/
│   └── db.py            # MongoDB client connection setup
├── models/
│   └── user.py          # Pydantic schema for User profile data
└── routes/
    ├── register.py      # /register & /check-user-id endpoints
    └── verify.py        # /verify (Liveness & DeepFace match) endpoint
```

---

## 🔄 Verification & Validation Code Flow

### 1. User Registration Flow (`POST /register`)

```
[ Upload Image + Form Data ] 
             │
             ▼
[ Decode Image (OpenCV) ] ──► (Fail? Return "Invalid image")
             │
             ▼
[ DeepFace Embedding ] ─────► Extract 128-d Facenet embedding vector
             │
             ▼
[ Duplicate Check ] ────────► Cosine similarity against all DB embeddings
                             └─► If similarity ≥ 40% (0.40), reject as duplicate
             │
             ▼
[ MongoDB Save ] ───────────► Save User profile + Binary image bytes
             │
             ▼
[ QR Code Creation ] ───────► Generate Base64 PNG QR Code containing user_id
```

1. **Input Payload**: `file` (Multipart image), `name` (Form string), `age` (Form int), `user_id` (Form string).
2. **Image Decoding**: Converts raw uploaded bytes into an OpenCV BGR image matrix (`cv2.imdecode`).
3. **Biometric Embedding**: Computes facial vector embedding using `DeepFace.represent(img, model_name="Facenet", detector_backend="opencv")`.
4. **Duplicate Prevention**: Iterates existing stored embeddings in MongoDB and computes Cosine Similarity ($\frac{\vec{v_1} \cdot \vec{v_2}}{\|\vec{v_1}\| \|\vec{v_2}\|}$). If similarity $\ge 0.40$, registration is rejected to prevent duplicate profiles.
5. **Persistence**: Saves profile metadata and binary image payload (`Binary(file_bytes)`) to MongoDB `users` collection.
6. **QR Generation**: Encodes `user_id` into a PNG QR code, returns it as a Base64 string.

---

### 2. Biometric Verification & Liveness Flow (`POST /verify`)

```
[ Upload Video/Photo + user_id ]
             │
             ▼
[ Motion Liveness Check ] ──► Frame-by-frame absdiff (detect_motion)
                             └─► Motion score < 0.5? Return ("is_live": False)
             │
             ▼
[ Fetch Reference Image ] ──► Retrieve stored BSON image from MongoDB by user_id
             │
             ▼
[ DeepFace Verification ] ──► DeepFace.verify(candidate_frame, stored_img)
                             ├─ Model: Facenet
                             └─ Metric: Cosine distance (d)
             │
             ▼
[ Threshold Evaluation ] ───► similarity = (1.0 - distance) * 100%
                             └─► distance ≤ 0.40 (similarity ≥ 60%)?
                                    ├─ YES ──► "verified": True + User Details
                                    └─ NO  ──► "verified": False + Error Message
```

1. **Input Payload**: `user_id` (Form string), `file` (Multipart video `.mp4` or photo `.jpg`).
2. **Motion-Based Liveness Detection**:
   - `detect_motion()` opens the file via `cv2.VideoCapture`.
   - Computes mean absolute frame difference (`cv2.absdiff`) across consecutive grayscale frames.
   - If multiple frames are present and peak movement `max_diff < 0.5`, liveness fails (`"is_live": False`). Single frame uploads pass by default to allow fallback photo testing.
3. **Database Reference Retrieval**: Queries MongoDB `users` collection for `user_id` and decodes `image_data` binary blob into OpenCV image matrix.
4. **Biometric Distance Calculation**:
   - Runs `DeepFace.verify()` with `Facenet` model and `cosine` distance metric.
   - Converts Cosine Distance ($d$) to similarity percentage: `similarity = max(0.0, 1.0 - distance) * 100.0`.
5. **Decision Logic**:
   - Verified if `distance <= 0.40` (equivalent to $\ge 60\%$ facial similarity threshold).
   - Returns JSON containing `verified` (boolean), `is_live` (boolean), `score_percent`, `threshold_percent`, and user profile details.

---

## ⚙️ How to Set Up & Run

### Prerequisites

* Python 3.11+
* MongoDB Instance (Local MongoDB or MongoDB Atlas URI)
* C++ Build tools / OpenCV dependencies (included in slim Docker image)

---

### Step 1: Clone Repository & Navigate to Backend

```bash
cd backend
```

### Step 2: Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/
```

---

### Step 5: Run Local Development Server

Start FastAPI with Uvicorn auto-reload:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
* **Interactive Docs (Swagger UI)**: `http://localhost:8000/docs`
* **Health Check**: `http://localhost:8000/health`

---

## 🐳 Docker Deployment

### Build Docker Image locally:
```bash
docker build -t face-recognition-backend .
```

### Run Container locally:
```bash
docker run -d -p 8000:8000 --env-file .env face-recognition-backend
```

---

## ☁️ Railway Deployment

1. Connect repository to Railway.
2. Set root directory to `backend/`.
3. Add environment variable `MONGO_URI` in Railway dashboard.
4. Railway will automatically build using the included `Dockerfile` and `railway.json`.
