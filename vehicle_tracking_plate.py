import cv2
import os
import re
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR


# ============================================================
# FILE / MODEL SETTINGS
# ============================================================

SAMPLE_IMAGE = "sample/vehicle_sample.jpg"

SAMPLE_VIDEO = "sample/vehicle_sample.mp4"

CAMERA_INDEX = 0

VEHICLE_MODEL_PATH = "yolo11n.pt"

PLATE_MODEL_PATH = "license-plate-finetune-v1n.pt"

SAVE_FILE = "detected_vehicles.txt"


# ============================================================
# VEHICLE SETTINGS
# ============================================================

VEHICLE_CONF = 0.35

PLATE_CONF = 0.15

VEHICLE_CLASSES = [2, 3, 5, 7]

CLASS_NAMES = {
    2: "CAR",
    3: "MOTORCYCLE",
    5: "BUS",
    7: "TRUCK"
}


# ============================================================
# UNIQUE PLATE MEMORY
# ============================================================

saved_plates = set()


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading vehicle model...")

vehicle_model = YOLO(
    VEHICLE_MODEL_PATH
)

print("Loading license plate model...")

plate_model = YOLO(
    PLATE_MODEL_PATH
)

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("All models loaded successfully!\n")


# ============================================================
# CLEAN PLATE TEXT
# ============================================================

def clean_plate_text(text):

    text = str(text).upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# SAVE VEHICLE NUMBER
# ============================================================

def save_plate(
    plate_number,
    vehicle_type
):

    global saved_plates

    if not plate_number:
        return

    plate_number = clean_plate_text(
        plate_number
    )

    if not plate_number:
        return

    # Don't save same number repeatedly
    if plate_number in saved_plates:
        return

    saved_plates.add(
        plate_number
    )

    now = datetime.now()

    date_time = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        SAVE_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{date_time} | "
            f"{plate_number} | "
            f"{vehicle_type}\n"
        )

    print(
        f"[SAVED] {plate_number} -> {vehicle_type}"
    )


# ============================================================
# OCR
# ============================================================

def read_plate(plate_crop):

    if plate_crop is None:
        return ""

    if plate_crop.size == 0:
        return ""

    try:

        height, width = plate_crop.shape[:2]

        if width < 200:

            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

        result = ocr.predict(
            plate_crop
        )

        best_text = ""

        best_score = 0

        if result is None:
            return ""

        for res in result:

            texts = []

            scores = []

            try:

                texts = res["rec_texts"]

                scores = res["rec_scores"]

            except Exception:

                try:

                    texts = getattr(
                        res,
                        "rec_texts",
                        []
                    )

                    scores = getattr(
                        res,
                        "rec_scores",
                        []
                    )

                except Exception:

                    texts = []

                    scores = []

            if texts is None:
                texts = []

            if scores is None:
                scores = []

            for i, text in enumerate(texts):

                text = clean_plate_text(
                    text
                )

                if not text:
                    continue

                try:

                    score = float(
                        scores[i]
                    )

                except Exception:

                    score = 0

                if score > best_score:

                    best_score = score

                    best_text = text

        if best_score >= 0.30:

            return best_text

        return ""

    except Exception as e:

        print(
            "OCR error:",
            e
        )

        return ""


# ============================================================
# POINT INSIDE VEHICLE
# ============================================================

def point_inside_box(
    point,
    box
):

    x, y = point

    x1, y1, x2, y2 = box

    return (
        x1 <= x <= x2
        and
        y1 <= y <= y2
    )


# ============================================================
# MATCH PLATE WITH VEHICLE
# ============================================================

def find_vehicle_for_plate(
    plate_box,
    vehicles
):

    px1, py1, px2, py2 = plate_box

    center_x = int(
        (px1 + px2) / 2
    )

    center_y = int(
        (py1 + py2) / 2
    )

    best_vehicle = None

    best_area = None

    for vehicle in vehicles:

        box = vehicle["box"]

        if point_inside_box(
            (center_x, center_y),
            box
        ):

            x1, y1, x2, y2 = box

            area = (
                (x2 - x1)
                *
                (y2 - y1)
            )

            if (
                best_area is None
                or
                area < best_area
            ):

                best_area = area

                best_vehicle = vehicle

    return best_vehicle


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(
    frame,
    tracking=False
):

    display_frame = frame.copy()

    vehicles = []

    # ========================================================
    # VEHICLE DETECTION
    # ========================================================

    if tracking:

        results = vehicle_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=VEHICLE_CLASSES,
            conf=VEHICLE_CONF,
            verbose=False
        )

    else:

        results = vehicle_model.predict(
            frame,
            classes=VEHICLE_CLASSES,
            conf=VEHICLE_CONF,
            verbose=False
        )

    if (
        results
        and
        len(results) > 0
    ):

        result = results[0]

        if result.boxes is not None:

            for index, box in enumerate(
                result.boxes
            ):

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                cls = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                vehicle_name = (
                    CLASS_NAMES.get(
                        cls,
                        "VEHICLE"
                    )
                )

                track_id = None

                if tracking:

                    if box.id is not None:

                        track_id = int(
                            box.id[0]
                        )

                if track_id is None:

                    track_id = index + 1

                vehicles.append({

                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),

                    "class_id": cls,

                    "name": vehicle_name,

                    "confidence": confidence,

                    "id": track_id,

                    "plate": ""

                })


    # ========================================================
    # PLATE DETECTION
    # ========================================================

    plate_results = plate_model.predict(
        frame,
        conf=PLATE_CONF,
        imgsz=480,
        verbose=False
    )

    detected_plates = []

    if (
        plate_results
        and
        len(plate_results) > 0
    ):

        plate_result = plate_results[0]

        if plate_result.boxes is not None:

            for plate_box in (
                plate_result.boxes
            ):

                xyxy = (
                    plate_box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                px1, py1, px2, py2 = map(
                    int,
                    xyxy
                )

                confidence = float(
                    plate_box.conf[0]
                )

                detected_plates.append({

                    "box": (
                        px1,
                        py1,
                        px2,
                        py2
                    ),

                    "confidence": confidence

                })


    # ========================================================
    # OCR
    # ========================================================

    for plate in detected_plates:

        px1, py1, px2, py2 = (
            plate["box"]
        )

        h, w = frame.shape[:2]

        px1 = max(
            0,
            px1
        )

        py1 = max(
            0,
            py1
        )

        px2 = min(
            w,
            px2
        )

        py2 = min(
            h,
            py2
        )

        if (
            px2 <= px1
            or
            py2 <= py1
        ):
            continue

        plate_crop = frame[
            py1:py2,
            px1:px2
        ]

        plate_text = read_plate(
            plate_crop
        )

        matched_vehicle = (
            find_vehicle_for_plate(
                (
                    px1,
                    py1,
                    px2,
                    py2
                ),
                vehicles
            )
        )

        # Plate box
        cv2.rectangle(
            display_frame,
            (px1, py1),
            (px2, py2),
            (255, 0, 0),
            2
        )

        if plate_text:

            label = plate_text

        else:

            label = "READING..."

        cv2.putText(
            display_frame,
            label,
            (
                px1,
                max(
                    25,
                    py1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        if matched_vehicle is not None:

            matched_vehicle[
                "plate"
            ] = plate_text


    # ========================================================
    # VEHICLE DISPLAY
    # ========================================================

    counts = {

        "CAR": 0,

        "MOTORCYCLE": 0,

        "BUS": 0,

        "TRUCK": 0

    }

    for vehicle in vehicles:

        x1, y1, x2, y2 = (
            vehicle["box"]
        )

        name = vehicle["name"]

        confidence = (
            vehicle["confidence"]
        )

        track_id = vehicle["id"]

        plate = vehicle["plate"]

        counts[name] = (
            counts.get(name, 0)
            + 1
        )

        # Vehicle box
        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        if plate:

            label = (
                f"{plate} -> {name}"
            )

            save_plate(
                plate,
                name
            )

        else:

            label = (
                f"ID {track_id} | "
                f"{name} | "
                f"{confidence:.2f}"
            )

        (
            tw,
            th
        ), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            display_frame,
            (
                x1,
                max(
                    0,
                    y1 - th - 10
                )
            ),
            (
                x1 + tw + 8,
                y1
            ),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            display_frame,
            label,
            (
                x1 + 4,
                y1 - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )


    # ========================================================
    # DASHBOARD
    # ========================================================

    cv2.rectangle(
        display_frame,
        (0, 0),
        (315, 195),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        display_frame,
        "VEHICLE MONITOR",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"CAR        : {counts['CAR']}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"MOTORCYCLE : {counts['MOTORCYCLE']}",
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"BUS        : {counts['BUS']}",
        (15, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"TRUCK      : {counts['TRUCK']}",
        (15, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    total = sum(
        counts.values()
    )

    cv2.putText(
        display_frame,
        f"TOTAL      : {total}",
        (15, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    return (
        display_frame,
        vehicles,
        counts,
        detected_plates
    )


# ============================================================
# SAMPLE IMAGE
# ============================================================

def run_sample_image():

    print("\n========================================")

    print(
        "SAMPLE IMAGE MODE"
    )

    print(
        "========================================"
    )

    if not os.path.exists(
        SAMPLE_IMAGE
    ):

        print(
            "\nERROR: Sample image not found!"
        )

        print(
            os.path.abspath(
                SAMPLE_IMAGE
            )
        )

        return

    frame = cv2.imread(
        SAMPLE_IMAGE
    )

    if frame is None:

        print(
            "ERROR: Cannot read sample image."
        )

        return

    print(
        f"Image: {SAMPLE_IMAGE}"
    )

    print(
        "Running detection...\n"
    )

    (
        processed_frame,
        vehicles,
        counts,
        plates
    ) = process_frame(
        frame,
        tracking=False
    )

    print(
        "\n========================================"
    )

    print(
        "SAMPLE IMAGE RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"CAR        : {counts['CAR']}"
    )

    print(
        f"MOTORCYCLE : {counts['MOTORCYCLE']}"
    )

    print(
        f"BUS        : {counts['BUS']}"
    )

    print(
        f"TRUCK      : {counts['TRUCK']}"
    )

    print(
        f"TOTAL      : {sum(counts.values())}"
    )

    print(
        f"PLATES DETECTED : {len(plates)}"
    )

    print(
        "========================================"
    )

    for vehicle in vehicles:

        if vehicle["plate"]:

            print(
                f"{vehicle['plate']} "
                f"-> {vehicle['name']}"
            )

        else:

            print(
                f"Plate not read -> "
                f"{vehicle['name']}"
            )

    print(
        "========================================"
    )

    print(
        "\nPress Q to close."
    )

    cv2.imshow(
        "Vehicle Tracking + ANPR",
        processed_frame
    )

    while True:

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    cv2.destroyAllWindows()


# ============================================================
# SAMPLE VIDEO
# ============================================================
def run_sample_video():

    print("\n========================================")
    print("SAMPLE VIDEO MODE")
    print("========================================")

    if not os.path.exists(SAMPLE_VIDEO):

        print("\nERROR: Sample video not found!")

        print(
            os.path.abspath(SAMPLE_VIDEO)
        )

        return

    cap = cv2.VideoCapture(
        SAMPLE_VIDEO
    )

    if not cap.isOpened():

        print(
            "ERROR: Cannot open sample video."
        )

        return

    print(
        f"Video: {SAMPLE_VIDEO}"
    )

    print(
        "Running video detection..."
    )

    print(
        "Press Q to stop."
    )

    frame_count = 0

    # Process AI only once every 10 frames
    FRAME_SKIP = 20

    last_result = None

    while True:

        ret, frame = cap.read()

        if not ret:

            print("\nVideo finished.")

            break

        frame_count += 1

        # Resize video
        frame = cv2.resize(
            frame,
            (640, 360)
        )

        # --------------------------------------------
        # AI PROCESSING
        # --------------------------------------------

        if frame_count % FRAME_SKIP == 0:

            (
                processed_frame,
                vehicles,
                counts,
                plates
            ) = process_frame(
                frame,
                tracking=True
            )

            last_result = processed_frame

        # --------------------------------------------
        # DISPLAY
        # --------------------------------------------

        if last_result is not None:

            cv2.imshow(
                "Vehicle Tracking + ANPR",
                last_result
            )

        else:

            cv2.imshow(
                "Vehicle Tracking + ANPR",
                frame
            )

        # --------------------------------------------
        # Q TO STOP
        # --------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    print(
        "\n========================================"
    )

    print(
        "SAMPLE VIDEO STOPPED"
    )

    print(
        "========================================"
    )
# ============================================================
# REAL-TIME WEBCAM
# ============================================================

def run_webcam():

    print("\n========================================")

    print(
        "REAL-TIME WEBCAM MODE"
    )

    print(
        "========================================"
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        print(
            "\nERROR: Webcam could not be opened."
        )

        print(
            "Try CAMERA_INDEX = 1"
        )

        return

    print(
        "Webcam started successfully."
    )

    print(
        "Press Q to stop."
    )

    print(
        f"\nVehicle numbers will be saved to:"
    )

    print(
        os.path.abspath(
            SAVE_FILE
        )
    )

    print(
        "========================================"
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Cannot read webcam."
            )

            break

        (
            processed_frame,
            vehicles,
            counts,
            plates
        ) = process_frame(
            frame,
            tracking=True
        )

        cv2.imshow(
            "Vehicle Tracking + ANPR",
            processed_frame
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    print(
        "\n========================================"
    )

    print(
        "WEBCAM STOPPED"
    )

    print(
        "========================================"
    )


# ============================================================
# INPUT MENU
# ============================================================

def show_menu():

    print("\n")

    print(
        "============================================"
    )

    print(
        " VEHICLE TRACKING + NUMBER PLATE RECOGNITION"
    )

    print(
        "============================================"
    )

    print(
        "\nChoose Input Source:"
    )

    print(
        "1. Sample Image"
    )

    print(
        "2. Sample Video"
    )

    print(
        "3. Real-Time Webcam"
    )

    print(
        "============================================"
    )

    while True:

        choice = input(
            "\nEnter your choice (1/2/3): "
        ).strip()

        if choice in ["1", "2", "3"]:

            return choice

        print(
            "Invalid choice! "
            "Please enter 1, 2, or 3."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    choice = show_menu()

    if choice == "1":

        run_sample_image()

    elif choice == "2":

        run_sample_video()

    elif choice == "3":

        run_webcam()