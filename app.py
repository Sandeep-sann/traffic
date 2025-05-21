import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("go.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detected_output.mp4", fourcc, 20.0, (1270, 960))

def get_stats(total, stopped):
    signal_time = 20 if total > 12 else (10 if total > 5 else 6)
    signal_color = (0, 0, 255) if stopped > 0 else (0, 255, 0)
    density_level = "High" if total > 12 else "Low"
    return signal_time, signal_color, density_level

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1270, 960))
    height, width = frame.shape[:2]

    # Background subtraction
    fgmask = fgbg.apply(frame)
    motion_mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    # Run YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    left_total = left_stopped = 0
    right_total = right_stopped = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        label = model.names[int(cls)]

        if label not in ['car', 'truck', 'bus', 'motorcycle']:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        region = motion_mask[y1:y2, x1:x2]
        motion_pixels = cv2.countNonZero(region)
        is_moving = motion_pixels > ((x2 - x1) * (y2 - y1)) * 0.1

        color = (0, 255, 0) if is_moving else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if cx < width // 2:
            left_total += 1
            left_stopped += not is_moving
        else:
            right_total += 1
            right_stopped += not is_moving

    # Signal logic
    left_time, left_color, left_density = get_stats(left_total, left_stopped)
    right_time, right_color, right_density = get_stats(right_total, right_stopped)

    # Left info
    cv2.circle(frame, (150, 80), 20, left_color, -1)
    cv2.putText(frame, f"Density: {left_density}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.putText(frame, f"Vehicle Count: {left_total}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    edge_density = round(np.count_nonzero(motion_mask[:, :width // 2]) / ((width // 2) * height) * 100, 2)
    cv2.putText(frame, f"Edge Density: {edge_density}%", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.putText(frame, f"Signal Time: {left_time} sec", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)

    # Right info
    cv2.circle(frame, (width - 170, 80), 20, right_color, -1)
    cv2.putText(frame, f"Density: {right_density}", (width - 300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    cv2.putText(frame, f"Vehicle Count: {right_total}", (width - 300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    edge_density = round(np.count_nonzero(motion_mask[:, width // 2:]) / ((width // 2) * height) * 100, 2)
    cv2.putText(frame, f"Edge Density: {edge_density}%", (width - 300, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    cv2.putText(frame, f"Signal Time: {right_time} sec", (width - 300, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)

    # Save and show frame
    out.write(frame)
    cv2.imshow("Smart Traffic System", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
