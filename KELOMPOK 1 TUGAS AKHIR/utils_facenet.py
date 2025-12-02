import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def face_align(bgr_img):
    """
    Input  : OpenCV image (BGR)
    Output : aligned face tensor (3x160x160) or None
    """
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    face_tensor = mtcnn(pil_img)
    return face_tensor

def embed_face_tensor(face_tensor):
    """
    Input  : aligned face tensor
    Output : 512D numpy embedding
    """
    if face_tensor is None:
        return None

    face_tensor = face_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy()[0]

    return emb
