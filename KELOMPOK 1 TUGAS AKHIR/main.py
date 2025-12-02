from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from utils_facenet import face_align, embed_face_tensor

import numpy as np
import cv2, tempfile, os
from datetime import datetime
from db import SessionLocal, FaceUser, init_db

app = FastAPI(title="FaceBank API")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PHOTOS_DIR = STATIC_DIR / "photos"

print(f"[INFO] BASE DIR   = {BASE_DIR}")
print(f"[INFO] STATIC DIR = {STATIC_DIR}")

PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def home_page():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/register-page")
def register_page():
    return FileResponse(str(STATIC_DIR / "register.html"))

@app.get("/dashboard")
def dashboard_page():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()
    print("[INFO] Database siap dipakai (facebank.db)")

@app.get("/api/status")
def api_status():
    db = SessionLocal()
    total = db.query(FaceUser).count()
    db.close()
    return {"message": "FaceBank API ready", "total_user": total}

@app.post("/register/{nama}")
async def register_face(nama: str, file: UploadFile = File(...)):
    content = await file.read()

    safe_name = nama.strip().replace(" ", "_")
    filename = f"{safe_name}_{int(datetime.utcnow().timestamp())}.jpg" 
    photo_path = PHOTOS_DIR / filename

    with open(photo_path, "wb") as f:
        f.write(content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)
    os.remove(tmp_path)

    if img is None:
        raise HTTPException(400, "File gambar rusak atau tidak valid")

    face_tensor = face_align(img)
    if face_tensor is None:
        if os.path.exists(photo_path):
            os.remove(photo_path)
        raise HTTPException(400, "Wajah tidak terdeteksi di foto ini")

    emb = embed_face_tensor(face_tensor)
    if emb is None:
        if os.path.exists(photo_path):
            os.remove(photo_path)
        raise HTTPException(400, "Gagal memproses embedding wajah")

    emb_bytes = emb.astype(np.float32).tobytes()
    now_time = datetime.utcnow()

    db = SessionLocal()
    
    existing = db.query(FaceUser).filter(FaceUser.nama == nama).first()

    if existing:
        if existing.photo_path and os.path.exists(existing.photo_path):
            try:
                os.remove(existing.photo_path)
            except:
                pass
        
        existing.embedding = emb_bytes
        existing.photo_path = f"static/photos/{filename}"
        existing.registered_at = now_time
    else:
        db.add(FaceUser(
            nama=nama,
            embedding=emb_bytes,
            photo_path=f"static/photos/{filename}",
            registered_at=now_time
        ))

    db.commit()
    total = db.query(FaceUser).count()
    db.close()

    return {"status": "Berhasil", "user": nama, "total_user": total}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    content = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    img = cv2.imread(tmp_path)
    os.remove(tmp_path)

    if img is None:
        raise HTTPException(400, "Gambar tidak valid")

    face_tensor = face_align(img)
    if face_tensor is None:
        raise HTTPException(400, "Wajah tidak terdeteksi")

    input_emb = embed_face_tensor(face_tensor)
    if input_emb is None:
        raise HTTPException(400, "Gagal embedding")

    db = SessionLocal()
    users = db.query(FaceUser).all()

    if len(users) == 0:
        db.close()
        return {"status": "empty", "message": "Database kosong"}

    best_name, best_dist = None, float("inf")

    for u in users:
        bank_emb = np.frombuffer(u.embedding, dtype=np.float32)
        dist = np.linalg.norm(input_emb - bank_emb)
        if dist < best_dist:
            best_dist = dist
            best_name = u.nama

    db.close()

    THRESHOLD = 0.8 
    
    if best_dist < THRESHOLD:
        return {"status": "match", "user": best_name, "distance": float(best_dist)}

    return {"status": "unknown", "distance": float(best_dist)}

@app.get("/users")
def list_users():
    db = SessionLocal()
    users = db.query(FaceUser.nama).all()
    db.close()
    return {"users": [u[0] for u in users]}

@app.get("/users/detail")
def list_users_detail():
    db = SessionLocal()
    users = db.query(FaceUser).all()
    db.close()

    return {
        "users": [
            {
                "nama": u.nama,
                "photo_path": u.photo_path,
                "registered_at": u.registered_at.isoformat()
            }
            for u in users
        ]
    }

@app.delete("/users/{nama}")
def delete_user(nama: str):
    db = SessionLocal()
    user = db.query(FaceUser).filter(FaceUser.nama == nama).first()

    if not user:
        db.close()
        raise HTTPException(404, "User tidak ditemukan")

    file_path = user.photo_path 

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"[INFO] File foto {file_path} berhasil dihapus.")
        except Exception as e:
            print(f"[ERROR] Gagal menghapus file: {e}")
    else:
        print(f"[WARNING] File {file_path} tidak ditemukan, hanya menghapus data DB.")

    db.delete(user)
    db.commit()
    
    total = db.query(FaceUser).count()
    db.close()

    return {"status": "Terhapus", "user": nama, "total_user": total}

@app.put("/users/{old_name}")
def update_user(old_name: str, new_name: str):
    db = SessionLocal()
    
    user = db.query(FaceUser).filter(FaceUser.nama == old_name).first()
    if not user:
        db.close()
        raise HTTPException(404, "User tidak ditemukan")

    if new_name != old_name:
        existing = db.query(FaceUser).filter(FaceUser.nama == new_name).first()
        if existing:
            db.close()
            raise HTTPException(400, "Nama baru sudah digunakan user lain")

    old_path = user.photo_path
    if old_path and os.path.exists(old_path):
        timestamp = int(datetime.utcnow().timestamp())
        safe_new_name = new_name.strip().replace(" ", "_")
        new_filename = f"{safe_new_name}_{timestamp}.jpg"
        new_path = PHOTOS_DIR / new_filename
        
        try:
            os.rename(old_path, new_path)
            user.photo_path = f"static/photos/{new_filename}" # Update path di DB
        except Exception as e:
            print(f"[WARNING] Gagal rename file: {e}")

    user.nama = new_name
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(500, "Gagal update database")

    db.close()
    return {"status": "Berhasil", "old_name": old_name, "new_name": new_name}