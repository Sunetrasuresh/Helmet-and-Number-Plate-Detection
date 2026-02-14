import cv2
import numpy as np
import os

# At the top
stop_detection = False

def stop_stream():
    global stop_detection
    stop_detection = True

# ==========================================================
# Load YOLOv3 configurations and weights - CORRECTED PATHS
# ==========================================================
# Get the helmetandnumber folder path (2 levels up from this file)
current_dir = os.path.dirname(os.path.abspath(__file__))  # detection folder
helmet_detection_dir = os.path.dirname(current_dir)  # helmet_detection folder
helmetandnumber_dir = os.path.dirname(helmet_detection_dir)  # helmetandnumber folder

# Construct paths to YOLO files
yolo_config_path = os.path.join(helmetandnumber_dir, "yolov3-custom.cfg")
yolo_weights_path = os.path.join(helmetandnumber_dir, "yolov3-custom_7000 (1).weights")
yolo_classes_path = os.path.join(helmetandnumber_dir, "classes.names")

# Print paths to verify (you'll see these in the console)
print(f"✓ Looking for YOLO files in: {helmetandnumber_dir}")
print(f"✓ Config path: {yolo_config_path}")
print(f"✓ Weights path: {yolo_weights_path}")
print(f"✓ Classes path: {yolo_classes_path}")

# Check if files exist before loading
if not os.path.exists(yolo_config_path):
    print(f"❌ ERROR: Config file not found at {yolo_config_path}")
    print("   Please check if yolov3-custom.cfg exists in the helmetandnumber folder")
    exit(1)

if not os.path.exists(yolo_weights_path):
    print(f"❌ ERROR: Weights file not found at {yolo_weights_path}")
    print("   Please check if yolov3-custom_7000 (1).weights exists")
    exit(1)

if not os.path.exists(yolo_classes_path):
    print(f"❌ ERROR: Classes file not found at {yolo_classes_path}")
    print("   Please check if classes.names exists")
    exit(1)

# Load class labels
with open(yolo_classes_path, "r") as f:
    classes = f.read().strip().split("\n")
print(f"✓ Loaded classes: {classes}")

# Load YOLOv3 network
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("✓ YOLO network loaded successfully!")

# ==========================================================
# Function: Process detections (your existing code continues here...)
# ==========================================================
def process_detections(outputs, width, height, frame, nms_threshold=0.4):
    helmet_detected = {}
    plates = []
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            label_text = classes[class_id]
            if label_text == "Helmet" and confidence < 0.75:
                label_text = "No Helmet"

            if label_text == "Helmet":
                color = (199, 150, 40)
            elif label_text == "No Helmet":
                color = (0, 0, 255)
            elif label_text == "Number_plate" or label_text == "Number-Plate":
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            label = f"{label_text} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_x = x
            label_y = y - 10

            text_bg_color = (0, 0, 0) if "Number" in label_text else color

            cv2.rectangle(frame,
                          (label_x, label_y - label_size[1]),
                          (label_x + label_size[0], label_y),
                          text_bg_color, -1)
            cv2.putText(frame, label, (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if label_text == "Helmet":
                helmet_detected[(x, y)] = "Helmet"
            elif label_text == "No Helmet":
                helmet_detected[(x, y)] = "No Helmet"
            elif "Number" in label_text:
                plates.append((x, y, x + w, y + h))

    return helmet_detected, plates


# ==========================================================
# Frame generator (for Django Streaming + local quit)
# ==========================================================
def generate_frames(video_path):
    global stop_detection
    stop_detection = False
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        return

    print(f"✓ Processing video: {video_path}")
    print("Press 'Q' in the OpenCV window to stop detection.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(output_layers)

        helmet_data, plate_data = process_detections(outputs, width, height, frame)

        for x, y, x_end, y_end in plate_data:
            cv2.putText(frame, "Number Plate", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.putText(frame, "Press Q to stop detection",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detection stopped by user (Q pressed).")
            break

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()
    cv2.destroyAllWindows()