import cv2
from deepface import DeepFace
import os
import sqlite3
import fire
import sqlite_vec
import struct
from typing import List


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def create_tables(conn, model_name):
    if model_name == "VGG-Face":
        dims = 4096
    else:
        dims = 128

    try:
        c = conn.cursor()
        # Create the cast table
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS cast (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """
        )
        # Create the virtual table for embeddings
        c.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vector_embeddings USING vec0(embedding float[{dims}])"
        )
    except Exception as e:
        print(e)


def insert_embedding(conn, name: str, embedding: List[float]):
    cur = conn.cursor()
    cur.execute("SELECT id FROM cast WHERE name = ?", (name,))
    cast_id = cur.fetchone()
    if cast_id is None:
        # Insert the name into the cast table if it doesn't exist
        cur.execute("INSERT INTO cast (name) VALUES (?)", (name,))
        cast_id = cur.lastrowid
    else:
        cast_id = cast_id[0]

    # Insert the embedding into the vector_embeddings table
    sql = """ INSERT INTO vector_embeddings(rowid, embedding)
              VALUES(?,?) """
    cur.execute(sql, (cast_id, serialize_f32(embedding)))
    conn.commit()


def extract_face_embeddings(image_path, model_name):
    img = cv2.imread(image_path)
    results = DeepFace.represent(
        img,
        model_name=model_name,
        detector_backend="yolov8",
        enforce_detection=True,
    )
    print(f"Extracted {len(results)} faces from image {image_path}")
    return results[0]["embedding"]


def extract_cast_embeddings(cast_dir, model_name="Facenet", db_path="cast.db"):
    print(f"Using model {model_name}, saving to db {db_path}")
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    create_tables(db, model_name)
    for file_name in os.listdir(cast_dir):
        if (
            file_name.endswith(".jpeg")
            or file_name.endswith(".jpg")
            or file_name.endswith(".png")
        ):
            name = os.path.splitext(file_name)[0]
            image_path = os.path.join(cast_dir, file_name)
            embedding = extract_face_embeddings(image_path, model_name)
            # Convert the embedding to bytes before inserting
            insert_embedding(db, name, embedding)
    db.close()


if __name__ == "__main__":
    fire.Fire(extract_cast_embeddings)
