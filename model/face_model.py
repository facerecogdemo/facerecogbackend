from pathlib import Path

import cv2
import numpy as np


MODEL_DIR = Path(__file__).resolve().parent

DETECTOR_PATH = MODEL_DIR / "face_detection_yunet.onnx"
RECOGNIZER_PATH = MODEL_DIR / "face_recognition_sface.onnx"


class FaceModel:
    def __init__(self):
        if not DETECTOR_PATH.exists():
            raise FileNotFoundError(
                f"Face detector model not found: {DETECTOR_PATH}"
            )

        if not RECOGNIZER_PATH.exists():
            raise FileNotFoundError(
                f"Face recognition model not found: {RECOGNIZER_PATH}"
            )

        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_PATH),
            "",
            (320, 320),
            0.8,
            0.3,
            5000,
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(RECOGNIZER_PATH),
            "",
        )

    def get_embedding(self, image_bytes):
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "Invalid image. Upload a JPG, PNG, or WEBP image."
            )

        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            raise ValueError(
                "No face detected. Adjust your position and try again."
            )

        if len(faces) != 1:
            raise ValueError(
                "Please upload an image containing exactly one face."
            )

        aligned_face = self.recognizer.alignCrop(image, faces[0])
        embedding = self.recognizer.feature(aligned_face)

        return embedding.flatten().astype(float).tolist()

    @staticmethod
    def similarity(first, second):
        first_vector = np.asarray(first, dtype=np.float32)
        second_vector = np.asarray(second, dtype=np.float32)

        denominator = (
            np.linalg.norm(first_vector)
            * np.linalg.norm(second_vector)
        )

        if denominator == 0:
            return -1.0

        return float(
            np.dot(first_vector, second_vector) / denominator
        )
