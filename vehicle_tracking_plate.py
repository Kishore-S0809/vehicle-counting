from ultralytics import YOLO
import cv2

# =========================
# SETTINGS
# =========================

# 0 = laptop webcam
# If you want to use input_video.mp4 later:
# SOURCE = "input_video.mp4"
SOURCE = 0

# YOLO model
MODEL = "yolo11n.pt"
PLATE_MODEL = "license-plate-finetune-v1n.pt"

# Vehicle classes in COCO
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
VEHICLE_CLASSES = [2, 3, 5, 7]


# =========================
# LOAD MODEL
# =========================

print("Loading YOLO model...")

model = YOLO(MODEL)
plate_model = YOLO(PLATE_MODEL)

print("Model loaded successfully!")
print("Starting vehicle tracking...")


# =========================
# VIDEO / WEBCAM
# =========================

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("ERROR: Cannot open camera/video.")
    exit()


# =========================
# TRACKING LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video ended or camera frame unavailable.")
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()
        # =========================
    # LICENSE PLATE DETECTION
    # =========================

    plate_results = plate_model.predict(
        frame,
        conf=0.35,
        verbose=False
    )

    for plate_result in plate_results:

        if plate_result.boxes is None:
            continue

        for plate_box in plate_result.boxes.xyxy.cpu().tolist():

            px1, py1, px2, py2 = map(int, plate_box)

            # Draw plate rectangle
            cv2.rectangle(
                annotated_frame,
                (px1, py1),
                (px2, py2),
                (255, 0, 0),
                2
            )

            # Plate label
            cv2.putText(
                annotated_frame,
                "LICENSE PLATE",
                (px1, max(py1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    # Get tracking boxes
    if results[0].boxes.id is not None:

        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            classes
        ):

            x1, y1, x2, y2 = map(int, box)

            # Vehicle name
            vehicle_name = model.names[class_id]

            # Display tracking ID
            text = f"ID: {track_id} | {vehicle_name}"

            cv2.putText(
                annotated_frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Show result
    cv2.imshow(
        "Live Vehicle Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================

cap.release()
cv2.destroyAllWindows()

print("Tracking stopped.")