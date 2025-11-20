# FaceNet untuk Verifikasi dan Identifikasi Wajah

## Tujuan Praktikum

- Memahami alur kerja FaceNet: deteksi ‘face alignment’ dan ekstraksi embedding
512-dim.
- Melakukan verifikasi wajah 1:1 berbasis kemiripan kosinus. 
- Melakukan identifikasi multi-orang menggunakan klasifier (SVM/KNN) di atas
embedding.
- Melakukan identifikasi wajah menggunakan SVM  
- MMengevaluasi akurasi dan menetapkan ambang (threshold) yang tepat.

  ## 📂 Struktur Folder
 FaceNet/
│
├── data/
│   ├── train/
│   │   ├── Iin/
│   │   │   ├── Iin1.jpg
│   │   │   └── Iin2.jpg
│   │   └── Lisa/
│   │       ├── Lisa1.jpg
│   │       └── Lisa2.jpg
│   │
│   └── val/
│       ├── Iin/
│       │   ├── Iin1.jpg
│       │   └── Iin2.jpg
│       └── Lisa/
│           ├── Lisa1.jpg
│           └── Lisa2.jpg
│
├── build_embeddings.py
├── eval_folder.py
├── facenet_svm.joblib
├── predict_one.py
├── train_classifier.py
├── train_knn.py
├── utils_facenet.py
├── verify_cli.py
├── verify_pair.py


## Analisis file kode
