# cli.py 
import argparse
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.ingest import (
    extract_exif,
    compute_image_embedding,
    extract_frames,
    scan_git_repo_for_secrets
)
from backend.app.faiss_manager import FaissManager

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
INDEX_PATH = os.path.join(DATA_DIR, 'index.faiss')
META_PATH = os.path.join(DATA_DIR, 'faiss_meta.json')
os.makedirs(DATA_DIR, exist_ok=True)

faiss_manager = FaissManager(dim=512)
if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    try:
        faiss_manager.load(INDEX_PATH, META_PATH)
    except Exception:
        print("[WARN] Could not load FAISS index; continuing with empty index.")

def cmd_image(path: str, persist: bool):
    print('[*] Processing image:', path)
    exif = extract_exif(path)
    emb = compute_image_embedding(path)
    print('[+] EXIF:')
    print(json.dumps(exif, indent=4))
    print('[+] Embedding length:', len(emb))
    if persist:
        try:
            if len(emb) != faiss_manager.dim:
                print(f"[WARN] Embedding dim {len(emb)} != FAISS dim {faiss_manager.dim}; not stored.")
            else:
                uid = faiss_manager.add(emb, metadata={"filename": os.path.basename(path)})
                faiss_manager.save(INDEX_PATH, META_PATH)
                print(f'[+] Stored in FAISS with ID: {uid}')
        except Exception as e:
            print("[ERROR] Failed to store embedding:", e)

def cmd_video(path: str):
    print('[*] Processing video:', path)
    tmpdir = tempfile.mkdtemp()
    try:
        frames = extract_frames(path, out_dir=tmpdir, fps=1)
        print('[+] Extracted frames:', len(frames))
        for f in frames[:3]:
            print('  frame:', f)
    except Exception as e:
        print("[ERROR] Video processing failed:", e)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def cmd_git(url: str):
    import git
    tmp = tempfile.mkdtemp()
    print('[*] Cloning:', url)
    try:
        git.Repo.clone_from(url, tmp)
        print("\n===== RUNNING SECRET SCAN =====\n")
        findings = scan_git_repo_for_secrets(tmp)
        for item in findings:
            print(item)
        print(f"\n✅ Total findings: {len(findings)}")
        with open("raw_scan.json", "w") as f:
            json.dump(findings, f, indent=4)
        print("\n📁 Report saved: raw_scan.json")
    except Exception as e:
        print("[ERROR] Git scan failed:", e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    p = argparse.ArgumentParser(description="PS1 Security CLI")
    p.add_argument('--image', help='Path to image')
    p.add_argument('--video', help='Path to video')
    p.add_argument('--git', help='Public git repo URL')
    p.add_argument('--persist', action='store_true', help='Store image embedding in FAISS')
    args = p.parse_args()
    if args.image:
        cmd_image(args.image, args.persist)
    elif args.video:
        cmd_video(args.video)
    elif args.git:
        cmd_git(args.git)
    else:
        p.print_help()

if __name__ == '__main__':
    main()
