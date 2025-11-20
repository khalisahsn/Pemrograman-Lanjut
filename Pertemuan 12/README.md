build_embeddings.py

import os, glob, numpy as np
from tqdm import tqdm
from utils_facenet import embed_from_path

def iter_images(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,
    d))])
    for cls in classes:
        for p in glob.glob(os.path.join(root, cls, "*")):
            yield p, cls

def build_matrix(root):
    X, y, bad = [], [], []
    for path, cls in tqdm(list(iter_images(root))):
        emb = embed_from_path(path)
        if emb is None:
            bad.append(path)
            continue
        X.append(emb); y.append(cls)
    return np.array(X), np.array(y), bad
if __name__ == "__main__":
    X, y, bad = build_matrix("data/train")
    print("Embeddings:", X.shape, "Labels:", y.shape, "Gagal deteksi:", len(bad))
    np.save("X_train.npy", X); np.save("y_train.npy", y)

#Analisis#
Script ini bertugas untuk menghasilkan embedding wajah dari seluruh gambar yang ada pada folder data/train. Pada bagian awal, modul seperti os, glob, dan numpy diimpor untuk membantu penanganan file dan manipulasi array, sedangkan tqdm digunakan untuk menampilkan progress bar agar proses terlihat lebih informatif. Fungsi embed_from_path dari utils_facenet dipanggil untuk mengambil embedding 512 dimensi dari setiap wajah yang berhasil terdeteksi.

Fungsi iter_images(root) bertugas menghasilkan pasangan (path, class) untuk setiap gambar yang ada dalam struktur folder data/train. Struktur data yang diharapkan adalah satu folder per identitas, sehingga fungsi ini mencatat nama folder sebagai label kelas. Setiap file dalam folder akan dikembalikan satu per satu tanpa memuat seluruh gambar langsung ke memori.

Fungsi utama build_matrix(root) melakukan iterasi terhadap semua gambar yang ditemukan oleh iter_images. Untuk setiap path gambar, fungsi mencoba menghasilkan embedding. Jika embedding gagal (misalnya wajah tidak terdeteksi), path tersebut dicatat dalam list bad. Jika berhasil, embedding dimasukkan ke list X, dan label kelas dimasukkan ke list y. Di akhir proses, embedding dikonversi ke array numpy agar lebih efisien untuk pemrosesan machine-learning selanjutnya.

Pada blok if __name__ == "__main__":, script mengeksekusi proses pembuatan embedding dan mencetak statistik seperti jumlah embedding dan jumlah gambar gagal deteksi. Hasil embedding dan label kemudian disimpan ke file X_train.npy dan y_train.npy, yang akan digunakan oleh script training classifier seperti SVM atau KNN.