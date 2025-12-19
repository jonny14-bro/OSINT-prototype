
🛡️ PS1 – OSINT Security Scanner

PS1 OSINT Security Scanner is a Python-based, multi-modal OSINT and security analysis framework that ingests images, videos, text, audio, and Git repositories to extract metadata, detect sensitive information, perform AI-based analysis, and enable similarity & semantic search using FAISS.

This project is designed for educational, research, and defensive security purposes and is tested on Kali Linux (Python 3.10).

🚀 Key Features
🔍 Image Analysis

EXIF & GPS metadata extraction

Image embedding generation (CLIP)

Image similarity search using FAISS

Duplicate detection

🎥 Video Analysis

Frame extraction using OpenCV

Image-to-video similarity matching

YOLOv8 object detection (mandatory)

Optional object detection during ingestion

🧠 Text OSINT

Email, phone number, username detection

Possible credential leakage detection

Semantic search using MiniLM embeddings + FAISS

🔊 Audio Analysis

Spectral feature extraction

Environment classification (indoor / outdoor)

Speech-to-text transcription (best effort)

🧬 Git Repository Scanning

Clone public Git repositories

Regex-based secret scanning

Severity-based clean reports (API keys, passwords, secrets)

📦 FAISS Vector Database

Separate indexes for:

Images / video frames (512-dim)

Text semantic search (384-dim)

Persistent on-disk storage

Similarity & semantic search

🔐 Admin Panel (CLI)

Password-protected admin login

View database statistics

Reset / wipe all stored data safely

⚠️ Mandatory Dependency: YOLOv8

This project REQUIRES YOLOv8 for object detection.

YOLO is used for detecting objects in video frames and images

Implemented using the ultralytics library

Runs on CPU by default (GPU optional)

❗ If YOLO is not installed, the program will fail.

🧰 Tech Stack

Python 3.10

FastAPI – backend API

FAISS – vector similarity search

Sentence-Transformers (CLIP & MiniLM)

YOLOv8 (Ultralytics)

OpenCV

Pillow / piexif

GitPython

Librosa & SpeechRecognition

📁 Project Structure
ps1-osint/
├── backend/
│   └── app/
│       ├── ingest.py          # Core OSINT logic
│       ├── main.py            # FastAPI routes
│       ├── faiss_manager.py   # FAISS wrapper
│       ├── faiss_registry.py  # Multi-index setup
├── scripts/
│   └── cli.py                 # CLI utility
├── UI.py                      # Interactive terminal UI
├── requirements.txt           # Full (YOLO + ML + FAISS)
├── requirements-lite.txt      # Lightweight demo
├── README.md
├── KALI-SETUP.md
└── instructions.md

🖥️ Installation (Kali Linux)
1️⃣ System Dependencies
sudo apt update
sudo apt install -y git ffmpeg libgl1 build-essential \
  python3.10 python3.10-venv python3.10-dev

2️⃣ Create Virtual Environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

3️⃣ Install Python Dependencies (FULL – REQUIRED)
pip install -r requirements.txt


⚠️ This installs YOLOv8, Torch, FAISS, and ML models
Installation may take time depending on system resources.

▶️ Running the Application
🔹 Start FastAPI Server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000


Check:

curl http://127.0.0.1:8000/health

🔹 Run Interactive CLI UI
python UI.py


Features available:

Image scan

Video scan (with YOLO)

Git repo scan

Audio scan

Text OSINT

Keyword & semantic search

Admin panel

📡 API Examples
Image Ingest
curl -F "file=@image.jpg" http://127.0.0.1:8000/ingest/image

Video Ingest with Object Detection
curl -F "file=@video.mp4" \
"http://127.0.0.1:8000/ingest/video?detect_objects=true"

Git Repository Scan
curl -X POST \
"http://127.0.0.1:8000/ingest/git?url=https://github.com/user/repo"

🔐 Security & Ethics Disclaimer

⚠️ WARNING

This tool is intended for educational, research, and defensive security use only.

Scan only data you own or have permission to analyze

Do NOT use on unauthorized systems or networks

Authors are not responsible for misuse

🎓 Academic Note

This project is suitable for:

BTech / Engineering projects

Cybersecurity & OSINT coursework

AI + ML + Security portfolios

It demonstrates:

Real-world OSINT workflows

Computer vision (YOLO + CLIP)

Semantic search

Secure data handling

📌 Future Enhancements

GPU acceleration

Face recognition (optional)

OSINT graph visualization

Web dashboard frontend

Encrypted database storage
