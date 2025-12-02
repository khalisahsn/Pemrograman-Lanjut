from sqlalchemy import create_engine, Column, String, LargeBinary, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DB_URL = "sqlite:///facebank.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class FaceUser(Base):
    __tablename__ = "face_users"

    nama = Column(String, primary_key=True, index=True)
    embedding = Column(LargeBinary, nullable=False)

    photo_path = Column(String, nullable=False)                 # path foto
    registered_at = Column(DateTime, default=datetime.utcnow)  # waktu daftar

def init_db():
    Base.metadata.create_all(bind=engine)
