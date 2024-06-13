import cv2
from deepface import DeepFace
import fire
import sqlite3
import sqlite_vec
import struct
from typing import List
import torch
from ultralytics import YOLO
from torchvision import transforms


dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2_vits14 = dinov2_vits14.eval()
model = YOLO("/Users/lunaticd/Downloads/yolov8n-face.pt")
# model = model.to("mps")
#

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
# transform(img).unsqueeze(dim=0)
# out = dinov2_vits14(inp).reshape(-1)
# out /= out.norm(dim=-1, keepdim=True)
# ret = out.detach().numpy()


def detect_faces(img, confidence=0.4):
    resp = []

    # Detect faces
    results = model.predict(img, verbose=False, show=False, conf=confidence)[0]

    # For each face, extract the bounding box, the landmarks and confidence
    for result in results:
        if result.boxes is None or result.keypoints is None:
            continue

        # Extract the bounding box and the confidence
        x, y, w, h = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]

        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)
        facial_area = dict(
            x=x,
            y=y,
            w=w,
            h=h,
            confidence=confidence,
        )
        resp.append(facial_area)

    return resp


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def extract_face_embeddings(image_path, model_name):
    img = cv2.imread(image_path)
    results = DeepFace.represent(
        img,
        model_name=model_name,
        detector_backend="yolov8",
    )
    print(f"Extracted {len(results)} faces from image {image_path}")
    return results[0]["embedding"]


class DB:
    def __init__(self, db_path):
        self.db_path = db_path
        db = sqlite3.connect(db_path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        self.db = db

    def find_closest_cast_member(self, query: List[float]):
        cur = self.db.cursor()
        q = """
    SELECT
        rowid,
        distance
    FROM vector_embeddings
    WHERE embedding MATCH ?
    ORDER BY distance
    LIMIT 1
    """
        row = cur.execute(q, [serialize_f32(query)]).fetchone()
        if not row:
            print("Couldn't find match")
            return None
        distance = row[1]
        row = cur.execute("SELECT name FROM cast WHERE id = ?", (row[0],)).fetchone()
        name = row[0]
        cur.close()
        return dict(name=name, distance=distance)


# Perform clustering on embeddings
# model_name="FaceNet",
def extract_faces_from_video(
    video_path,
    model_name="Facenet",
    db_path="cast.db",
    capture_every_n_frames=5,
):
    print(f"Capturing faces in video {video_path}")
    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    db = DB(db_path)

    fc = 0
    while True:
        ret, frame = cap.read()
        fc += 1
        if not ret:
            break

        if fc % capture_every_n_frames != 0:
            continue

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Detect faces in the frame
        # results = DeepFace.represent(
        #     frame,
        #     detector_backend="yolov8",
        #     model_name=model_name,
        #     enforce_detection=False,
        # )
        results = detect_faces(frame)

        for r in results:
            # embedding = r["embedding"]
            # facial_area = r["facial_area"]
            facial_area = r

            # Draw rectangle around the face
            x, y, w, h = (
                facial_area["x"],
                facial_area["y"],
                facial_area["w"],
                facial_area["h"],
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # found = db.find_closest_cast_member(embedding)
            # if found is not None and found["distance"] < 11.4:
            #     print(f"Found cast member {found} at timestmap {timestamp}")

        cv2.imshow("Video", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release video capture object and close display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire(extract_faces_from_video)
