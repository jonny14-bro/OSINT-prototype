import os
import tempfile
import shutil
import re
from typing import Optional, List

from PIL import Image
import piexif
import imagehash

# ------------------ MODEL CACHES ------------------
_CLIP_MODEL = None
_TEXT_MODEL = None
_YOLO_MODEL = None


def _load_clip():
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _CLIP_MODEL = SentenceTransformer('clip-ViT-B-32')
        except Exception:
            _CLIP_MODEL = None
    return _CLIP_MODEL


def _load_text_model():
    global _TEXT_MODEL
    if _TEXT_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _TEXT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            _TEXT_MODEL = None
    return _TEXT_MODEL


def _load_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception:
            _YOLO_MODEL = None
    return _YOLO_MODEL


# ------------------ EXIF ------------------
def extract_exif(path: str) -> dict:
    result = {}
    try:
        img = Image.open(path)
        exif_data = img._getexif() or {}

        try:
            exif_ifd = piexif.load(path)
            result['piexif'] = exif_ifd
        except:
            result['piexif'] = None

        result['raw_exif'] = exif_data
        gps = _extract_gps(result['piexif'], exif_data)
        if gps:
            result['gps'] = gps

    except Exception as e:
        result['error'] = str(e)
    return result


def _extract_gps(piexif_data, raw_exif) -> Optional[dict]:
    try:
        if piexif_data and 'GPS' in piexif_data:
            gps = piexif_data['GPS']
            if not gps:
                return None

            def _to_deg(v):
                return v[0][0]/v[0][1] + v[1][0]/v[1][1]/60 + v[2][0]/v[2][1]/3600

            lat = _to_deg(gps.get(piexif.GPSIFD.GPSLatitude))
            lon = _to_deg(gps.get(piexif.GPSIFD.GPSLongitude))

            lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef, b'N').decode()
            lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef, b'E').decode()

            if lat_ref != 'N': lat = -lat
            if lon_ref != 'E': lon = -lon

            return {"lat": lat, "lon": lon}
    except:
        pass
    return None


# ------------------ IMAGE EMBEDDING ------------------
def compute_image_embedding(path: str) -> List[float]:
    model = _load_clip()
    if model is not None:
        try:
            img = Image.open(path).convert('RGB')
            return model.encode(img, convert_to_numpy=True).tolist()
        except:
            pass

    try:
        phash = imagehash.phash(Image.open(path))
        h = int(str(phash), 16)
        return [(h >> i) & 0xFFFF for i in range(0, 128, 16)]
    except:
        return []


# ------------------ VIDEO FRAMES ------------------
def extract_frames(video_path: str, out_dir: Optional[str] = None, fps: int = 1) -> List[str]:
    import cv2

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix='frames_')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps_video / fps))

    saved = []
    idx = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % step == 0:
            path = os.path.join(out_dir, f"frame_{count:05d}.jpg")
            cv2.imwrite(path, frame)
            saved.append(path)
            count += 1
        idx += 1

    cap.release()
    return saved


# ------------------ SECRET SCANNER ------------------
def scan_git_repo_for_secrets(repo_path: str) -> List[dict]:
    findings = []
    patterns = [
        r'API[_-]?KEY\s*[=:]\s*["\']?.{16,}["\']?',
        r'SECRET[_-]?KEY\s*[=:]\s*["\']?.{8,}["\']?',
        r'password\s*[=:]\s*["\']?.{6,}["\']?'
    ]

    compiled = [re.compile(p, re.I) for p in patterns]

    for root, _, files in os.walk(repo_path):
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, 'r', errors='ignore') as fh:
                    data = fh.read()
                    for p in compiled:
                        for m in p.finditer(data):
                            findings.append({"file": path, "match": m.group(0)})
            except:
                pass
    return findings


def clean_secret_findings(raw, repo_url=None):
    clean = []
    for item in raw:
        fp = item.get("file", "").lower()
        if any(x in fp for x in ["node_modules", ".min.js", "dist"]):
            continue

        txt = item["match"].lower()
        if "password" in txt:
            typ, sev = "Password", "High"
        elif "api" in txt:
            typ, sev = "API Key", "Critical"
        elif "secret" in txt:
            typ, sev = "Secret Key", "Critical"
        else:
            continue

        clean.append({
            "repo": repo_url,
            "file": item["file"],
            "type": typ,
            "leak": item["match"][:120],
            "severity": sev,
        })
    return clean


def scan_git_repo_for_secrets_with_reports(repo_path, repo_url=None):
    raw = scan_git_repo_for_secrets(repo_path)
    return {"raw_findings": raw, "clean_report": clean_secret_findings(raw, repo_url)}


# ------------------ YOLO OBJECT DETECTION ------------------
def detect_objects_in_image(image_path, conf_thresh=0.25):
    model = _load_yolo()
    if model is None:
        return [{"error": "YOLO not available"}]

    results = model(image_path)
    detections = []

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue

            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": label,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })

    return detections


# ------------------ TEXT OSINT ------------------
def analyze_text_osint(text: str) -> dict:
    return {
        "emails": re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text),
        "phones": re.findall(r"\+?\d[\d -]{8,12}\d", text),
        "usernames": re.findall(r"@[\w_]+", text),
        "possible_credentials": re.findall(r"(password|passwd|pwd)[\s:=]+[\S]+", text, re.I)
    }


def compute_text_embedding(text: str) -> list:
    model = _load_text_model()
    if model:
        return model.encode(text).tolist()
    return []


# ------------------ AUDIO ------------------
def analyze_audio(path: str) -> dict:
    result = {"transcript": None, "environment": [], "spectral_features": {}}
    try:
        import librosa, numpy as np, speech_recognition as sr

        y, sr_rate = librosa.load(path, sr=None)
        result["spectral_features"] = {
            "spectral_centroid": float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_rate))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y)))
        }

        env = "Urban / Traffic Noise" if result["spectral_features"]["spectral_centroid"] > 2500 else "Indoor / Quiet"
        result["environment"].append(env)

        rec = sr.Recognizer()
        with sr.AudioFile(path) as src:
            audio = rec.record(src)
            try:
                result["transcript"] = rec.recognize_google(audio)
            except:
                pass
    except Exception as e:
        result["error"] = str(e)

    return result
