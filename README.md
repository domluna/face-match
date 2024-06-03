# Face Detection Demo

This demo showcases a face detection and recognition system using the `deepface` library, OpenCV, and SQLite for database management. The system detects faces in videos and matches them against a database of known faces. If a match with a face in the database is sufficiently close, the system will output a confirmation.

## Package Dependencies

- `deepface`
- `opencv`
- `sqlite3`
- `sqlite_vec`

## Step-by-Step Guide

### 1. Building the Face Database

Before detecting and recognizing faces in videos, you need to build a database of known faces. This database will be used for matching detected faces. The `build_db.py` script helps you achieve this.

```sh
python build_db.py cast-photos --model_name Facenet --db_path cast.db
```

- **`build_db.py`**: This script processes a directory of face images to create a database.
- **`cast-photos`**: Directory containing images of faces to be included in the database.
- **`--model_name Facenet`**: Specifies the model used for facial recognition. `Facenet` is a powerful deep learning model for face recognition.
- **`--db_path cast.db`**: Specifies the path where the database will be stored.

The script performs the following tasks:
1. Loads images from the specified directory.
2. Uses the specified model to extract facial features from these images.
3. Stores these features in a SQLite database for quick retrieval during recognition.

### 2. Recognizing Faces in Videos

After building the database, you can use the `recognize.py` script to detect and recognize faces in a video.

```sh
python recognize.py vids/video.webm --model_name Facenet --db_path cast.db
```

- **`recognize.py`**: This script processes a video file to detect and recognize faces.
- **`vids/video.webm`**: Path to the video file where faces need to be detected.
- **`--model_name Facenet`**: Specifies the model used for facial recognition.
- **`--db_path cast.db`**: Specifies the path to the database containing known faces.

The script performs the following tasks:
1. Opens the specified video file.
2. Uses OpenCV to detect faces in each frame of the video.
3. For each detected face, extracts facial features using the specified model.
4. Compares the extracted features against the features stored in the database.
5. If the distance between the detected face features and the closest match in the database is below a threshold (11.4), it outputs that a match was made.

### Additional Details

- **Threshold for Matching**: The script uses a threshold of 11.4 to determine if a detected face matches a face in the database. This value is chosen based on the characteristics of the `Facenet` model and can be adjusted for different models or accuracy requirements.
- **Visual Indication**: The script visually indicates detected faces in the video. Even if a face is detected, it might not match any face in the database, in which case no match output will be provided.
