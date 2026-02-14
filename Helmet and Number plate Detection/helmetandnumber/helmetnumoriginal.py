import cv2
import numpy as np

# Load YOLOv3 configurations and weights
yolo_config_path = r"C:\Helmet and Number plate Detection\helmetandnumber\yolov3-custom.cfg"  # Path to YOLOv3 config file
yolo_weights_path = r"C:\Helmet and Number plate Detection\helmetandnumber\yolov3-custom_7000 (1).weights"  # Path to YOLOv3 weights file
yolo_classes_path = r"C:\Helmet and Number plate Detection\helmetandnumber\classes.names"  # Path to classes file [helmet, number plate]

# Load class labels
with open(yolo_classes_path, "r") as f:
    classes = f.read().strip().split("\n")

# Load YOLOv3 network
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
net.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CPU
)  # Use DNN_TARGET_CUDA for GPU acceleration if available


# Define function to process detections
def process_detections(outputs, width, height, frame, nms_threshold=0.4):
    helmet_detected = {}
    plates = []

    boxes = []
    confidences = []
    class_ids = []

    # Parse outputs
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")

                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                # Append detection data
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, nms_threshold)

    if len(indices) > 0:  # Ensure there are valid indices
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]

            # Override label if confidence < 0.75
            label_text = classes[class_id]
            if label_text == "Helmet" and confidence < 0.75:
                label_text = "No Helmet"  # Override to "no-helmet" if confidence is low

            # Choose bounding box color
            if label_text == "Helmet":
                color = (199, 150, 40)  # Aero for helmet
            elif label_text == "No Helmet":
                color = (0, 0, 255)  # Red for no-helmet
            elif label_text == "Number-Plate":
                color = (255, 0, 0)  # Blue for number plate
            else:
                color = (255, 0, 0)  # White for unknown classes

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Add label text with offset to avoid overlap
            label = f"{label_text} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_x = x
            label_y = y - 10  # Adjust label placement

            # Adjust text background for number plates
            text_bg_color = (
                (0, 0, 0) if label_text == "Number-Plate" else color
            )  # Black for number plate text

            # Draw label with background for readability
            cv2.rectangle(
                frame,
                (label_x, label_y - label_size[1]),
                (label_x + label_size[0], label_y),
                text_bg_color,
                -1,
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            # Save detected helmet or plates
            if label_text == "Helmet":
                helmet_detected[(x, y)] = "Helmet"
            elif label_text == "No Helmet":
                helmet_detected[(x, y)] = "No Helmet"
            elif label_text == "Number-Plate":
                plates.append((x, y, x + w, y + h))

    return helmet_detected, plates


# Video path
video_path = (
    r"C:\Helmet and Number plate Detection\helmetandnumber\Helmet_videos\complete.mp4"
)


# Video input and processing
def detect_helmet_and_plate_yolov3(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the file can be opened
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        return

    # Processing each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Prepare the image for YOLO
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)

        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        # Perform forward pass
        outputs = net.forward(output_layers)

        # Process detections and annotates te frameS
        helmet_data, plate_data = process_detections(outputs, width, height, frame)

        # Highlight number plates
        for x, y, x_end, y_end in plate_data:
            cv2.putText(
                frame,
                "Number Plate",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        # Display the frame with detections
        cv2.imshow("Helmet and Number Plate Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()  # Releases the video file
    cv2.destroyAllWindows()  # Closes OpenCV dispay windows


# Run the detection
detect_helmet_and_plate_yolov3(video_path)
