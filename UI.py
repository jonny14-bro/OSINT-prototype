import os
import sys
import json
import tempfile
import shutil
import hashlib
import getpass


RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
CYAN = "\033[96m"
RESET = "\033[0m"

# -----------------------------------------------------
# PATH FIX FOR BACKEND IMPORTS
# -----------------------------------------------------
sys.path.append(os.path.abspath("."))

# -----------------------------------------------------
# PATH FIX FOR BACKEND IMPORTS
# -----------------------------------------------------
sys.path.append(os.path.abspath("."))

from backend.app.ingest import (
    extract_exif,
    compute_image_embedding,
    extract_frames,
    scan_git_repo_for_secrets_with_reports,
)
from backend.app.faiss_manager import FaissManager  # 512-dim manager for images/video frames
# ingest.py already defines compute_text_embedding/analyze_text_osint/analyze_audio :contentReference[oaicite:0]{index=0}


# -----------------------------------------------------
# FAISS + DATA DIR INITIALIZATION
# -----------------------------------------------------

# 512-dim FAISS for images + video frames
faiss_manager = FaissManager(dim=512)  # :contentReference[oaicite:1]{index=1}
# 384-dim FAISS for text semantic search
text_faiss = FaissManager(dim=384)     # separate index for MiniLM text embeddings

# Data directory and index/metadata paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "faiss_meta.json")

TEXT_INDEX_PATH = os.path.join(DATA_DIR, "text_index.faiss")
TEXT_META_PATH = os.path.join(DATA_DIR, "text_faiss_meta.json")

OSINT_DB_PATH = os.path.join(DATA_DIR, "osint_records.json")
ADMIN_CONFIG_PATH = os.path.join(DATA_DIR, "admin_config.json")


def _load_faiss_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
        try:
            faiss_manager.load(INDEX_PATH, META_PATH)
            print(f"[INFO] Loaded image/video FAISS index with {faiss_manager.count()} items.")
        except Exception as e:
            print(f"[WARN] Could not load FAISS index: {e}")
    else:
        print("[INFO] No existing image/video FAISS index found; starting empty.")


def _save_faiss_index():
    try:
        faiss_manager.save(INDEX_PATH, META_PATH)
    except Exception as e:
        print(f"[WARN] Failed to save FAISS index: {e}")


def _load_text_faiss_index():
    if os.path.exists(TEXT_INDEX_PATH) and os.path.exists(TEXT_META_PATH):
        try:
            text_faiss.load(TEXT_INDEX_PATH, TEXT_META_PATH)
            print(f"[INFO] Loaded text FAISS index with {text_faiss.count()} items.")
        except Exception as e:
            print(f"[WARN] Could not load text FAISS index: {e}")
    else:
        print("[INFO] No existing text FAISS index found; starting empty.")


def _save_text_faiss_index():
    try:
        text_faiss.save(TEXT_INDEX_PATH, TEXT_META_PATH)
    except Exception as e:
        print(f"[WARN] Failed to save text FAISS index: {e}")


def _load_osint_db():
    if not os.path.exists(OSINT_DB_PATH):
        return []
    try:
        with open(OSINT_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_osint_db(records):
    try:
        with open(OSINT_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
    except Exception as e:
        print(f"[WARN] Failed to save OSINT DB: {e}")


def _append_osint_record(record: dict):
    records = _load_osint_db()
    records.append(record)
    _save_osint_db(records)




# -----------------------------------------------------
# ADMIN / AUTH HELPERS
# -----------------------------------------------------
def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()


def _load_admin_config():
    if not os.path.exists(ADMIN_CONFIG_PATH):
        return None
    try:
        with open(ADMIN_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_admin_config(cfg: dict):
    try:
        with open(ADMIN_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        print(f"[WARN] Failed to save admin config: {e}")


def _ensure_admin_setup():
    cfg = _load_admin_config()
    if cfg is not None:
        return

    print("\n[ADMIN SETUP] No admin configured yet. Create an admin account.")
    username = input("Set admin username: ").strip()
    while not username:
        username = input("Username cannot be empty. Set admin username: ").strip()

    while True:
        pw1 = getpass.getpass("Set admin password: ")
        pw2 = getpass.getpass("Confirm admin password: ")
        if pw1 != pw2:
            print("[ERROR] Passwords do not match. Try again.")
        elif not pw1:
            print("[ERROR] Password cannot be empty.")
        else:
            break

    cfg = {
        "username": username,
        "password_hash": _hash_password(pw1),
    }
    _save_admin_config(cfg)
    print("[ADMIN] Admin account created successfully.\n")


def _admin_login() -> bool:
    _ensure_admin_setup()
    cfg = _load_admin_config()
    if cfg is None:
        print("[FATAL] Admin config missing or corrupted.")
        return False

    print("\n[ADMIN LOGIN]")
    user = input("Username: ").strip()
    pw = getpass.getpass("Password: ")

    if user != cfg.get("username"):
        print("[ERROR] Invalid admin username.")
        return False

    if _hash_password(pw) != cfg.get("password_hash"):
        print("[ERROR] Invalid admin password.")
        return False

    print("[INFO] Admin login successful.\n")
    return True


def _admin_reset_database():
    global faiss_manager, text_faiss

    confirm = input("This will DELETE all FAISS indexes and OSINT records. Type 'DELETE' to confirm: ").strip()
    if confirm != "DELETE":
        print("[INFO] Reset aborted.")
        return

    # Remove index/meta/db files
    for p in [INDEX_PATH, META_PATH, TEXT_INDEX_PATH, TEXT_META_PATH, OSINT_DB_PATH]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

    # Reinitialize managers
    faiss_manager = FaissManager(dim=512)
    text_faiss = FaissManager(dim=384)

    print("[INFO] Database reset complete. All embeddings and OSINT records cleared.")


def _admin_show_stats():
    records = _load_osint_db()
    total_records = len(records)
    img = sum(1 for r in records if r.get("type") == "image")
    vid = sum(1 for r in records if r.get("type") == "video")
    git = sum(1 for r in records if r.get("type") == "git")
    txt = sum(1 for r in records if r.get("type") == "text")
    aud = sum(1 for r in records if r.get("type") == "audio")

    print("\n[DB STATS]")
    print(f"  Total OSINT records : {total_records}")
    print(f"   - Images           : {img}")
    print(f"   - Videos           : {vid}")
    print(f"   - Git repos        : {git}")
    print(f"   - Text files       : {txt}")
    print(f"   - Audio files      : {aud}")
    print(f"  FAISS (images/video frames) count : {faiss_manager.count()}")
    print(f"  FAISS (text semantic) count      : {text_faiss.count()}")
    print()


def admin_panel_ui():
    if not _admin_login():
        return

    while True:
        print("\n[ADMIN PANEL]")
        print("1. Show database stats")
        print("2. Reset / clear all data")
        print("3. Back to main menu")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            _admin_show_stats()
        elif choice == "2":
            _admin_reset_database()
        elif choice == "3":
            break
        else:
            print("[ERROR] Invalid choice.")


# -----------------------------------------------------
# BANNER
# -----------------------------------------------------
def banner():
    print(RED + BOLD +    """⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣷⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠭⣿⣿⣿⣶⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣴⣾⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡿⣿⡿⣿⣿⣿⣿⣦⣴⣶⣶⣶⣶⣦⣤⣤⣀⣀⠀⠀⠀⠀⠀⢀⣀⣤⣲⣿⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⡝⢿⣌⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣤⣾⣿⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠲⡝⡷⣮⣝⣻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣛⣿⣿⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣦⣝⠓⠭⣿⡿⢿⣿⣿⣛⠻⣿⠿⠿⣿⣿⣿⣿⣿⣿⡿⣇⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣤⡀⠈⠉⠚⠺⣿⠯⢽⣿⣷⣄⣶⣷⢾⣿⣯⣾⣿⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⡟⠀⠀⣴⣿⣿⣼⠈⠉⠃⠋⢹⠁⢀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢿⣿⡟⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⣀⣀⣀⣀⣴⣿⣿⡿⣿⠀⠀⠀⠀⠇⠀⣼⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⢿⢿⣾⣿⣿⡿⠿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⠟⠿⢿⡄⢦⣤⣤⣶⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠘⠛⠋⠁⠁⣀⢉⡉⢻⡻⣯⣻⣿⢻⣿⣀⠀⠀⠀⢠⣾⣿⣿⣿⣹⠉⣍⢁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠠⠔⠒⠋⠀⡈⠀⠠⠤⠀⠓⠯⣟⣻⣻⠿⠛⠁⠀⠀⠣⢽⣿⡻⠿⠋⠰⠤⣀⡈⠒⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠔⠊⠁⠀⣀⠔⠈⠁⠀⠀⠀⠀⠀⣶⠂⠀⠀⠀⢰⠆⠀⠀⠀⠈⠒⢦⡀⠉⠢⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠊⠀⠀⠀⠀⠎⠁⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠰⠃⠀⠀⠀⠀⠀⠀⠀⠈⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠿⠭⠯⠭⠽⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
    """ + RESET)


def banner2():
    print(
        BOLD +
        """          
                    ██████╗ """ + RED + """███████╗""" + RESET + BOLD + """██╗███╗   ██╗████████╗
                   ██╔═══██╗""" + RED + """██╔════╝""" + RESET + BOLD + """██║████╗  ██║╚══██╔══╝
                   ██║   ██║""" + RED + """███████╗""" + RESET + BOLD + """██║██╔██╗ ██║   ██║   
                   ██║   ██║""" + RED + """╚════██║""" + RESET + BOLD + """██║██║╚██╗██║   ██║   
                   ╚██████╔╝""" + RED + """███████║""" + RESET + BOLD + """██║██║ ╚████║   ██║   
                    ╚═════╝ """ + RED + """╚══════╝""" + RESET + BOLD + """╚═╝╚═╝  ╚═══╝   ╚═╝   


                      ☠☠☠  D A N G E R   Z O N E  ☠☠☠
                     UNAUTHORIZED ACCESS = TERMINATION

                           PS1 SECURITY SCANNER
    """ + RESET
    )
    

    
    
# -----------------------------------------------------
# SAFE JSON CONVERTER
# -----------------------------------------------------
def safe_json(obj):
    try:
        from PIL.TiffImagePlugin import IFDRational
    except Exception:
        IFDRational = ()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, IFDRational):
        return float(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode(errors="ignore")
    if isinstance(obj, dict):
        return {safe_json(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_json(i) for i in obj]
    try:
        return str(obj)
    except Exception:
        return repr(obj)


# -----------------------------------------------------
# IMAGE SCAN
# -----------------------------------------------------
def scan_image_ui():
    path = input("Enter image file path: ").strip()
    if not os.path.exists(path):
        print("[ERROR] File does not exist."); return
    print("[*] Scanning image...")
    try:
        exif = extract_exif(path)
        clean_exif = safe_json(exif)
        if "piexif" in clean_exif and isinstance(clean_exif["piexif"], dict) and "thumbnail" in clean_exif["piexif"]:
            clean_exif["piexif"]["thumbnail"] = "[REMOVED]"

        embedding = compute_image_embedding(path)
        embedding_len = len(embedding) if embedding else 0

        uid = None
        if embedding and embedding_len == faiss_manager.dim:
            uid = faiss_manager.add(
                embedding,
                metadata={
                    "type": "image",
                    "filename": os.path.basename(path),
                    "source": path,
                    "exif_present": bool(clean_exif.get("raw_exif"))
                }
            )
            _save_faiss_index()

        # Console report
        print("[RESULT] Image Analysis:")
        print(json.dumps({"filename": os.path.basename(path), "exif": clean_exif, "embedding_len": embedding_len}, indent=4))

        # Similarity search (against previously stored items: images + video frames)
        if uid:
            matches = faiss_manager.search_by_uid(uid, k=5)
        else:
            matches = []

        if matches:
            print("[INFO] Similar items found:")
            for rid, dist, meta in matches:
                mtype = meta.get("type", "unknown")
                fname = meta.get("filename") or meta.get("frame") or "N/A"
                src = meta.get("source")
                print(f"  - [{mtype}] {fname} (source: {src}) (distance: {dist:.6f})")
        else:
            print("[INFO] No similar items found.")

        # Store OSINT record in JSON DB
        record = {
            "type": "image",
            "source": path,
            "filename": os.path.basename(path),
            "faiss_uid": uid,
            "exif": clean_exif,
            "embedding_len": embedding_len,
        }
        _append_osint_record(record)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")


# -----------------------------------------------------
# VIDEO SCAN  (stores frames in FAISS + image-to-video similarity)
# -----------------------------------------------------
def scan_video_ui():
    path = input("Enter video file path: ").strip()
    if not os.path.exists(path):
        print("[ERROR] File does not exist."); return
    print("[*] Scanning video frames...")
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    try:
        frames = extract_frames(path, out_dir=tmpdir, fps=1)
        print(f"[RESULT] Total frames extracted: {len(frames)}")

        sample_info = []
        max_frames_to_store = 20  # avoid exploding FAISS in very long videos

        for idx, f in enumerate(frames):
            emb = compute_image_embedding(f)
            emb_len = len(emb) if emb else 0

            # Collect sample info for first few frames
            if idx < 5:
                sample_info.append({
                    "frame": os.path.basename(f),
                    "embedding_len": emb_len
                })

            if emb and emb_len == faiss_manager.dim and idx < max_frames_to_store:
                # Add frame embedding to FAISS
                uid = faiss_manager.add(
                    emb,
                    metadata={
                        "type": "video_frame",
                        "filename": os.path.basename(f),
                        "source": path,
                        "frame_index": idx
                    }
                )

                # Optional: search for similar stored images for this frame
                matches = faiss_manager.search_by_uid(uid, k=3)
                if matches:
                    print(f"[FRAME MATCH] frame_{idx:05d}.jpg similar to:")
                    for rid, dist, meta in matches:
                        if meta.get("type") == "image":  # highlight similar images
                            print(f"   - image {meta.get('filename')} (source: {meta.get('source')}) dist={dist:.4f}")

        _save_faiss_index()

        print("[INFO] Sample frame embeddings:")
        print(json.dumps(sample_info, indent=4))

        # Store a summarised OSINT record for the video
        record = {
            "type": "video",
            "source": path,
            "num_frames": len(frames),
            "sample_frames": sample_info
        }
        _append_osint_record(record)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)



# -----------------------------------------------------
# GIT SECRET SCAN
# -----------------------------------------------------
def scan_git_ui():
    url = input("Enter Git repository URL: ").strip()
    if not url:
        print("[ERROR] URL cannot be empty."); return
    tmpdir = tempfile.mkdtemp(prefix="git_")
    try:
        print("[*] Cloning and scanning repository...")
        import git
        repo = git.Repo.clone_from(url, tmpdir)
        from backend.app.ingest import scan_git_repo_for_secrets_with_reports  # ensure we use same logic :contentReference[oaicite:2]{index=2}
        report = scan_git_repo_for_secrets_with_reports(tmpdir, repo_url=url)

        print("\n[Raw Findings]")
        if report["raw_findings"]:
            print(json.dumps(report["raw_findings"], indent=4))
        else:
            print("No raw findings.")

        severity_order = ["Critical", "High", "Medium", "Low", "Info"]
        organized = {sev: [] for sev in severity_order}
        for item in report.get("clean_report", []):
            sev = item.get("severity", "Info")
            if sev not in organized:
                organized[sev] = []
            organized[sev].append(item)

        print("\n[Clean Report - Organized by Severity]\n")
        any_found = False
        for sev in severity_order:
            items = organized.get(sev, [])
            if items:
                any_found = True
                print(f"--- {sev} ---")
                for it in items:
                    print(f"{it.get('file')} -> {it.get('type')}: {it.get('leak')}")
                print()
        if not any_found:
            print("No secrets found in clean report.")

        # Store OSINT record for this repo
        record = {
            "type": "git",
            "source": url,
            "raw_findings_count": len(report.get("raw_findings", [])),
            "clean_report": report.get("clean_report", [])
        }
        _append_osint_record(record)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# -----------------------------------------------------
# TEXT SCAN (now also stores text embedding for semantic search)
# -----------------------------------------------------
def scan_text_ui():
    file_path = input("Paste file dir for public text / bio / article content:\n").strip()

    if not os.path.exists(file_path):
        print("[ERROR] File not found!")
        return

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return

    if not text.strip():
        print("[WARN] File is empty.")
        return

    from backend.app.ingest import analyze_text_osint, compute_text_embedding  # :contentReference[oaicite:3]{index=3}

    findings = analyze_text_osint(text)
    emb = compute_text_embedding(text)

    print("\n[OSINT TEXT FINDINGS]")
    print(json.dumps(findings, indent=4))
    print("\n[Embedding Length]:", len(emb))

    # Store findings in JSON DB
    record = {
        "type": "text",
        "source": file_path,
        "findings": findings,
        "text_preview": text[:400]
    }
    _append_osint_record(record)

    # Store semantic embedding in 384-dim text FAISS
    if emb and len(emb) == text_faiss.dim:
        text_faiss.add(
            emb,
            metadata={
                "type": "text",
                "source": file_path,
                "findings": findings
            }
        )
        _save_text_faiss_index()
        print("[INFO] Text embedding stored in semantic search index.")
    elif emb:
        print("[INFO] Text embedding computed but dim mismatch; not stored in FAISS.")


# -----------------------------------------------------
# AUDIO SCAN
# -----------------------------------------------------
def scan_audio_ui():
    path = input("Enter audio file path: ").strip()
    if not os.path.exists(path):
        print("[ERROR] File does not exist."); return
    print("[*] Analyzing audio...]")
    try:
        from backend.app.ingest import analyze_audio  # :contentReference[oaicite:4]{index=4}
        result = analyze_audio(path)
        print(json.dumps(result, indent=4))

        record = {
            "type": "audio",
            "source": path,
            "analysis": result
        }
        _append_osint_record(record)

    except Exception as e:
        print(f"[FATAL ERROR] {e}")


# -----------------------------------------------------
# KEYWORD-BASED SEARCH (existing behavior)
# -----------------------------------------------------
def search_database_ui():
    query = input("Enter search keyword or phrase: ").strip()
    if not query:
        print("[ERROR] Empty query.")
        return
    q = query.lower()

    records = _load_osint_db()
    if not records:
        print("[INFO] Database is empty. Run some scans first.")
        return

    print(f'\n[SEARCH] Looking for "{query}" in stored OSINT records...\n')
    matches = []
    for rec in records:
        blob = json.dumps(rec, ensure_ascii=False).lower()
        if q in blob:
            matches.append(rec)

    if not matches:
        print("[INFO] No matches found.")
        return

    print(f"[INFO] Found {len(matches)} matching records:\n")
    for idx, rec in enumerate(matches, start=1):
        r_type = rec.get("type", "unknown")
        src = rec.get("source", "unknown")
        print(f"{idx}. ({r_type.upper()}) source = {src}")

        if r_type == "git":
            leaks = [it.get("leak", "") for it in rec.get("clean_report", [])]
            if leaks:
                print("   Leaks:")
                for leak in leaks[:3]:
                    print(f"      - {leak[:200]}")
        elif r_type == "text":
            findings = rec.get("findings", {})
            creds = findings.get("possible_credentials", [])
            emails = findings.get("emails", [])
            if emails:
                print("   Emails:")
                for e in emails[:3]:
                    print(f"      - {e}")
            if creds:
                print("   Possible credentials:")
                for c in creds[:3]:
                    print(f"      - {c[:200]}")
        elif r_type == "image":
            exif = rec.get("exif", {})
            if exif.get("gps"):
                print(f"   GPS: {exif['gps']}")
            print(f"   Embedding length: {rec.get('embedding_len')}")
        elif r_type == "video":
            print(f"   Frames: {rec.get('num_frames')} (sampled {len(rec.get('sample_frames', []))})")
        elif r_type == "audio":
            env = rec.get("analysis", {}).get("environment", [])
            print(f"   Environment guess: {', '.join(env) if env else 'N/A'}")

        print()


# -----------------------------------------------------
# SEMANTIC SEARCH (embedding-based, for TEXT)
# -----------------------------------------------------
def semantic_search_ui():
    query = input("Enter semantic search query (text): ").strip()
    if not query:
        print("[ERROR] Empty query.")
        return

    from backend.app.ingest import compute_text_embedding  # :contentReference[oaicite:5]{index=5}
    emb = compute_text_embedding(query)
    if not emb or len(emb) != text_faiss.dim:
        print("[ERROR] Could not compute valid text embedding for this query.")
        return

    results = text_faiss.search_by_vector(emb, k=10)
    if not results:
        print("[INFO] No semantic matches found in text index.")
        return

    print(f'\n[SEMANTIC SEARCH] Results for "{query}":\n')
    for idx, (uid, dist, meta) in enumerate(results, start=1):
        src = meta.get("source", "unknown")
        findings = meta.get("findings", {})
        preview = ""
        creds = findings.get("possible_credentials", [])
        emails = findings.get("emails", [])
        if emails:
            preview += "Emails: " + ", ".join(emails[:3]) + " | "
        if creds:
            preview += "Creds: " + " ; ".join(c[:60] for c in creds[:2]) + " | "

        print(f"{idx}. source = {src}")
        print(f"   distance = {dist:.4f}")
        if preview:
            print(f"   {preview}")
        print()


# -----------------------------------------------------
# MAIN MENU
# -----------------------------------------------------
def main_ui():
    while True:

        print(YELLOW + BOLD + "\nSelect a function:" + RESET)
        print("\n")
        print("1. Scan Image")
        print("2. Scan Video")
        print("3. Scan Git Repository")
        print("4. Scan Audio")
        print("5. Scan Text File (.txt)")
        print("6. Search Database (keyword)")
        print("7. Semantic Search (text)")
        print("8. Admin Panel")
        print("9. Exit")
        print(CYAN + BOLD + "run command=   OSINT --help ( to demonstrate how to use )" + RESET)
        print("\n")
        choice = input("Enter choice: ").strip()

        if not choice:
            continue   # Ignore empty buffered input

        if choice == "1":
            scan_image_ui()
        elif choice == "2":
            scan_video_ui()
        elif choice == "3":
            scan_git_ui()
        elif choice == "4":
            scan_audio_ui()
        elif choice == "5":
            scan_text_ui()
        elif choice == "6":
            search_database_ui()
        elif choice == "7":
            semantic_search_ui()
        elif choice == "8":
            admin_panel_ui()
        elif choice == "9":
            print("[INFO] Exiting...")
            break
        else:
            print("[ERROR] Invalid choice.")


# -----------------------------------------------------
# PROGRAM ENTRY
# -----------------------------------------------------
if __name__ == "__main__":
    _load_faiss_index()
    _load_text_faiss_index()
    banner()
    banner2()
    print( YELLOW + BOLD + "<< ⚠️  CAUTION ! DO NOT USE ON UNAUTHERISED NETWORKS OR SYSTEMS >>" + RESET)
    main_ui()
