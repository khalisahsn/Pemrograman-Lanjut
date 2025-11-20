import argparse
from utils_facenet import embed_from_path, cosine_similarity

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("img1")
    ap.add_argument("img2")
    ap.add_argument("--th", type=float, default=0.85)
    args = ap.parse_args()

    e1 = embed_from_path(args.img1)
    e2 = embed_from_path(args.img2)

    if e1 is None or e2 is None:
        print("Wajah tidak terdeteksi pada salah satu gambar.")
    else:
        sim = cosine_similarity(e1, e2)
        print(
            f"Similarity={sim:.4f} -> "
            f"{'MATCH' if sim >= args.th else 'NO MATCH'} (th={args.th})"
        )
