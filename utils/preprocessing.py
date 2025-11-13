import cv2
import numpy as np

IMG_SIZE = 64
MAX_LEN = 20

def preprocess_video(video_path, max_len=MAX_LEN, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        frames.append(resized)
    cap.release()

    if len(frames) < 8:
        raise ValueError("❌ Video does not have enough frames (minimum 8).")

    if len(frames) < max_len:
        frames += [np.zeros((img_size, img_size)) for _ in range(max_len - len(frames))]
    else:
        frames = frames[:max_len]

    frames = np.array(frames).reshape(max_len, img_size, img_size, 1).astype('float32') / 255.0
    return np.expand_dims(frames, axis=0)
