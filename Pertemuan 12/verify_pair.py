from utils_facenet import embed_from_path, cosine_similarity

img1 = "samples/Lisa1.jpg"
img2 = "samples/Lisa2.jpg"

emb1 = embed_from_path(img1)
emb2 = embed_from_path(img2)
if emb1 is None or emb2 is None:
    print("Wajah tidak terdeteksi pada salah satu gambar.")
else:
    sim = cosine_similarity(emb1, emb2)
    print("Cosine similarity:", sim)
    
    threshold = 0.85
    print("Match?" , "YA" if sim >= threshold else "TIDAK")