import os
import tempfile
import shutil
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional

from .ingest import (
    extract_exif,
    compute_image_embedding,
    extract_frames,
    scan_git_repo_for_secrets_with_reports,
    detect_objects_in_image,
    detect_landmarks_in_image
)
from .faiss_manager import FaissManager

app = FastAPI(title='PS1 OSINT Prototype')

# Respect environment variables to avoid heavy allocations at startup
FAISS_USE_HNSW = os.getenv('FAISS_USE_HNSW', 'false').lower() in ('1', 'true', 'yes')
from .faiss_registry import text_faiss, vision_faiss


# (same FAISS load/save/autosave code as before...)
# For brevity reuse existing implementation from your main.py — keep the same _load_faiss_on_startup, _save_faiss_on_shutdown, helpers.

# ... copy relevant FAISS startup/shutdown and helper functions here if needed ...
# For simplicity in this snippet assume the rest of main.py remains same as before, only ingest/video and ingest/git endpoints are updated.

@app.post('/ingest/video')
async def ingest_video(file: UploadFile = File(...),
                       detect_objects: bool = Query(False, description='Run object detection on sample frames'),
                       landmarks_dir: Optional[str] = Query(None, description='Path to local landmarks templates folder (optional)')):
    tmpdir = tempfile.mkdtemp(prefix='upload_')
    try:
        path = os.path.join(tmpdir, file.filename)
        with open(path, 'wb') as fh:
            fh.write(await file.read())

        frames = extract_frames(path, out_dir=os.path.join(tmpdir, 'frames'), fps=1)
        frame_info = []
        for f in frames[:10]:  # limit sample frames processed to 10
            emb = compute_image_embedding(f)
            info = {'frame': os.path.basename(f), 'embedding_len': len(emb) if emb else 0}

            if detect_objects:
                try:
                    info['objects'] = detect_objects_in_image(f, conf_thresh=0.25)
                except Exception as e:
                    info['objects_error'] = str(e)

            if landmarks_dir:
                try:
                    info['landmarks'] = detect_landmarks_in_image(f, landmarks_dir=landmarks_dir, top_k=3)
                except Exception as e:
                    info['landmarks_error'] = str(e)

            frame_info.append(info)

        return JSONResponse({'filename': file.filename, 'num_frames': len(frames), 'sample_frames': frame_info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@app.post('/ingest/git')
async def ingest_git(url: str):
    import git
    tmpdir = tempfile.mkdtemp(prefix='git_')
    try:
        repo = git.Repo.clone_from(url, tmpdir)
        report = scan_git_repo_for_secrets_with_reports(tmpdir, repo_url=url)
        return JSONResponse(report)
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)




@app.post('/ingest/text')
async def ingest_text(text: str = Query(...), persist: bool = False):
    from .ingest import analyze_text_osint, compute_text_embedding

    osint = analyze_text_osint(text)
    emb = compute_text_embedding(text)

    result = {
        "osint_findings": osint,
        "embedding_len": len(emb)
    }

    if persist and emb:
        if len(emb) != text_faiss.dim:
            result["warning"] = f"Embedding dim {len(emb)} != FAISS dim {text_faiss.dim}. Not stored."
        else:
            uid = text_faiss.add(emb, metadata={
                "type": "text",
                "osint_findings": osint
            })
            result["id"] = uid

    return JSONResponse(result)



    
    
    
    
@app.post('/ingest/audio')
async def ingest_audio(file: UploadFile = File(...)):
    tmpdir = tempfile.mkdtemp(prefix='audio_')
    try:
        path = os.path.join(tmpdir, file.filename)
        with open(path, 'wb') as fh:
            fh.write(await file.read())

        from .ingest import analyze_audio

        analysis = analyze_audio(path)
        return JSONResponse({
            "filename": file.filename,
            "audio_intelligence": analysis
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
