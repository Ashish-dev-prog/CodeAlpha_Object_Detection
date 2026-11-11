import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11s.pt"
CAMERA_INDEX = 1
IMAGE_SIZE = 960
GPU = 0

def main():
    model = YOLO(MODEL_PATH)

    # Print the model's class IDs for our test objects.
    targets = {"book", "bed", "chair"}
    print("\nTarget class IDs:")
    for class_id, name in model.names.items():
        if name.lower() in targets:
            print(f"{name}: {class_id}")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index.")

    # Request higher camera resolution.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nActual camera resolution: {width}x{height}")

    window = "YOLO11s Confidence Tuner"
    cv2.namedWindow(window)
    cv2.createTrackbar(
        "Confidence (%)", window, 20, 80, lambda value: None
    )

    print("\nControls: Adjust confidence slider | S = save | Q = quit")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            confidence = max(
                0.01,
                cv2.getTrackbarPos("Confidence (%)", window) / 100
            )

            results = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=confidence,
                device=GPU,
                verbose=False
            )

            annotated = results[0].plot()

            cv2.putText(
                annotated,
                f"Confidence: {confidence:.2f} | S: Save | Q: Quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(window, annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s"):
                cv2.imwrite("tuning_raw.jpg", frame)
                cv2.imwrite("tuning_result.jpg", annotated)
                print("Saved tuning_raw.jpg and tuning_result.jpg")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()