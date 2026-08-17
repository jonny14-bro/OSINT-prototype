
# 🛡️ PS1 – OSINT Security Scanner..

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![FAISS](https://img.shields.io/badge/FAISS-Semantic_Search-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Required-critical?logo=yolo)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Multi-modal OSINT & Security Analysis Framework**  
> Ingests **images, videos, text, audio, and Git repositories** for metadata extraction, sensitive info detection, AI-based analysis, and semantic search using FAISS.  
> ⚠️ Tested on **Kali Linux (Python 3.10)**.  
> Built for **educational, research, and defensive security purposes**.

---

## 🚀 Features

### 🔍 Image Analysis
- EXIF & GPS metadata extraction  
- CLIP embeddings + FAISS similarity search  
- Duplicate detection  

### 🎥 Video Analysis
- Frame extraction (OpenCV)  
- Image-to-video similarity matching  
- **YOLOv8 object detection (mandatory)**  

### 🧠 Text OSINT
- Email, phone number, username detection  
- Credential leakage detection  
- Semantic search (MiniLM + FAISS)  

### 🔊 Audio Analysis
- Spectral feature extraction  
- Indoor/outdoor classification  
- Speech-to-text transcription  

### 🧬 Git Repository Scanning
- Clone public repos  
- Regex-based secret scanning  
- Severity-based clean reports  

### 📦 FAISS Vector Database
- Separate indexes for images, video frames, text  
- Persistent on-disk storage  
- Similarity & semantic search  

### 🔐 Admin Panel (CLI)
- Password-protected login  
- Database statistics  
- Safe reset/wipe  

---

## ⚠️ Mandatory Dependency: YOLOv8
- Required for object detection in images & videos  
- Implemented via **Ultralytics YOLOv8**  
- Runs on CPU by default (GPU optional)  
- ❗ Without YOLO, the program will fail  

---

## 🧰 Tech Stack
- Python 3.10  
- FastAPI  
- FAISS  
- Sentence-Transformers (CLIP & MiniLM)  
- YOLOv8 (Ultralytics)  
- OpenCV, Pillow, piexif  
- GitPython  
- Librosa & SpeechRecognition  

---

## 📁 Project Structure
```
ps1-osint/
├── backend/app/
│   ├── ingest.py          # Core OSINT logic
│   ├── main.py            # FastAPI routes
│   ├── faiss_manager.py   # FAISS wrapper
│   ├── faiss_registry.py  # Multi-index setup
├── scripts/cli.py         # CLI utility
├── UI.py                  # Interactive terminal UI
├── requirements.txt       # Full (YOLO + ML + FAISS)
├── requirements-lite.txt  # Lightweight demo
├── README.md
├── KALI-SETUP.md
└── instructions.md
```

---

## 🖥️ Installation (Kali Linux)

### 1️⃣ System Dependencies
```bash
sudo apt update
sudo apt install -y git ffmpeg libgl1 build-essential \
  python3.10 python3.10-venv python3.10-dev
```

### 2️⃣ Virtual Environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3️⃣ Python Dependencies
```bash
pip install -r requirements.txt
```

⚠️ Installs YOLOv8, Torch, FAISS, and ML models.  
Installation may take time depending on system resources.

---

## ▶️ Running the Application

### FastAPI Server
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
Check:
```bash
curl http://127.0.0.1:8000/health
```

### CLI UI
```bash
python UI.py
```

Features: Image scan, Video scan (YOLO), Git repo scan, Audio scan, Text OSINT, Semantic search, Admin panel.

---

## 📡 API Examples

**Image Ingest**
```bash
curl -F "file=@image.jpg" http://127.0.0.1:8000/ingest/image
```

**Video Ingest with Object Detection**
```bash
curl -F "file=@video.mp4" \
"http://127.0.0.1:8000/ingest/video?detect_objects=true"
```

**Git Repository Scan**
```bash
curl -X POST \
"http://127.0.0.1:8000/ingest/git?url=https://github.com/user/repo"
```

---

## 🔐 Security & Ethics Disclaimer

⚠️ **WARNING**  
This tool is intended for **educational, research, and defensive security use only**.  
- Scan only data you own or have permission to analyze  
- Do **NOT** use on unauthorized systems or networks  
- Authors are not responsible for misuse  

---

## 🎓 Academic Note
Ideal for:
- BTech / Engineering projects  
- Cybersecurity & OSINT coursework  
- AI + ML + Security portfolios  

Demonstrates:
- Real-world OSINT workflows  
- Computer vision (YOLO + CLIP)  
- Semantic search  
- Secure data handling  

---

## 📌 Future Enhancements
- GPU acceleration  
- Face recognition (optional)  
- OSINT graph visualization  
- Web dashboard frontend  
- Encrypted database storage  

