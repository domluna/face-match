import cv2
from deepface import DeepFace
import fire
import sqlite3
import sqlite_vec
import struct
from typing import List, Dict
import collections


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def extract_face_embeddings(frame, model_name):
    results = DeepFace.represent(
        frame,
        model_name=model_name,
        detector_backend="yolov8",
        enforce_detection=False,
    )
    return results


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
            return None
        distance = row[1]
        row = cur.execute("SELECT name FROM cast WHERE id = ?", (row[0],)).fetchone()
        name = row[0]
        cur.close()
        return dict(name=name, distance=distance)


def process_frame(frame, timestamp, model_name, db):
    results = extract_face_embeddings(frame, model_name)
    found_names = []
    for r in results:
        embedding = r["embedding"]
        found = db.find_closest_cast_member(embedding)
        if found is not None and found["distance"] < 11.4:
            # Format timestamp as m:ss
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            formatted_timestamp = f"{minutes}:{seconds:02}"
            found_names.append((found["name"], formatted_timestamp))
    return found_names


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
    frame_count = 0
    results = collections.defaultdict(set)

    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break

        if frame_count % capture_every_n_frames != 0:
            continue

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        found_names = process_frame(frame, timestamp, model_name, db)
        for name, timestamp in found_names:
            results[name].add(timestamp)

    cap.release()

    for name, timestamps in results.items():
        print(f"{name} => {list(timestamps)}")


if __name__ == "__main__":
    fire.Fire(extract_faces_from_video)
