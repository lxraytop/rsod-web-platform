import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
from PIL import Image
import cv2
from app.config import settings
from app.models.schemas import DetectionBox, DetectionResult
from app.utils.file_utils import get_file_url


class DetectionService:
    def __init__(self):
        self.model = None
        self.class_names = {}
        self._load_model()
        self._init_class_names()

    def _load_model(self):
        if os.path.exists(settings.YOLO_MODEL_PATH):
            self.model = YOLO(settings.YOLO_MODEL_PATH)
        else:
            raise FileNotFoundError(f"Model file not found: {settings.YOLO_MODEL_PATH}")

    def _init_class_names(self):
        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def detect_single_image(self, image_path: str, model_name: str = "pest-v1") -> DetectionResult:
        start_time = time.time()
        detection_id = str(uuid.uuid4())

        results = self.model.predict(
            source=image_path,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            save=False
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, f"class_{class_id}")

                boxes.append(DetectionBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                ))

        result_filename = f"result_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(settings.RESULT_DIR, result_filename)

        annotated_image = results[0].plot()
        cv2.imwrite(result_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        detection_time = time.time() - start_time

        image_filename = os.path.basename(image_path)

        return DetectionResult(
            detection_id=detection_id,
            image_url=get_file_url(image_filename, "static/uploads"),
            result_image_url=get_file_url(result_filename, "static/results"),
            boxes=boxes,
            total_objects=len(boxes),
            detection_time=round(detection_time, 3),
            model_name=model_name,
            created_at=datetime.now()
        )


detection_service = DetectionService()
