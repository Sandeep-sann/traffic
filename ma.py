import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("go.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

# Output config
output_width, output_height = 960, 720
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("detected_output.mp4", fourcc, 20.0, (output_width, output_height))

# Skip points for ignoring detection areas
skip_points = []

# Fullscreen window setup
cv2.namedWindow("Smart Traffic System", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Smart Traffic System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Smart Traffic System", lambda e, x, y, f, p: skip_points.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

# Signal timer logic
signal_timer_right = 90
last_update = time.time()

# Main loop
frozen_frame_right = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (output_width, output_height))
    height, width = frame.shape[:2]
    fgmask = fgbg.apply(frame)
    motion_mask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    results = model(frame, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    left_total = left_stopped = 0
    right_total = right_stopped = 0
    center_margin = 80
    center_x_start = width // 2 - center_margin
    center_x_end = width // 2 + center_margin

    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        label = model.names[int(cls)]

        if label not in ['car', 'truck', 'bus', 'motorcycle']:
            continue

        if any(x1 <= px <= x2 and y1 <= py <= y2 for (px, py) in skip_points):
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if center_x_start <= cx <= center_x_end:
            continue

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

    # Always green for left side
    signal_timer_left = 999
    left_color = (0, 255, 0)

    # Update right timer
    current_time = time.time()
    if current_time - last_update >= 1:
        signal_timer_right = max(0, signal_timer_right - 1)
        last_update = current_time
        if signal_timer_right == 0:
            signal_timer_right = 90

    # Set right signal color
    if signal_timer_right > 60:
        right_color = (0, 255, 0)
    elif signal_timer_right > 30:
        right_color = (0, 255, 255)
    else:
        right_color = (0, 0, 255)

    # Save frozen frame on red signal
    if signal_timer_right <= 30:
        if frozen_frame_right is None:
            frozen_frame_right = frame.copy()
        frame[:, width // 2:] = frozen_frame_right[:, width // 2:]
        overlay = frame.copy()
        alpha = 0.5
        cv2.rectangle(overlay, (width // 2, 0), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        # "RIGHT PAUSED" text removed
    else:
        frozen_frame_right = None

    # Density levels
    def classify_density(count):
        if count >= 13:
            return "High"
        elif count >= 6:
            return "Medium"
        elif count >= 1:
            return "Low"
        else:
            return "None"

    left_density = classify_density(left_total)
    right_density = classify_density(right_total)

    edge_density_left = round(np.count_nonzero(motion_mask[:, :width // 2]) / ((width // 2) * height) * 100, 2)
    edge_density_right = round(np.count_nonzero(motion_mask[:, width // 2:]) / ((width // 2) * height) * 100, 2)

    # Left signal info
    cv2.putText(frame, f"LEFT SIGNAL", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv2.putText(frame, f"[GREEN]", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 2)
    cv2.circle(frame, (150, 100), 20, left_color, -1)
    cv2.putText(frame, f"Density: {left_density}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv2.putText(frame, f"Vehicle Count: {left_total}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
    cv2.putText(frame, f"Edge Density: {edge_density_left}%", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)

    # Right signal info
    cv2.putText(frame, f"RIGHT SIGNAL", (width - 280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    cv2.putText(frame, f"[{signal_timer_right}s]", (width - 280, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 2)
    cv2.circle(frame, (width - 170, 100), 20, right_color, -1)
    cv2.putText(frame, f"Density: {right_density}", (width - 300, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    cv2.putText(frame, f"Vehicle Count: {right_total}", (width - 300, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
    cv2.putText(frame, f"Edge Density: {edge_density_right}%", (width - 300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)

    # Draw skipped areas
    for px, py in skip_points:
        cv2.circle(frame, (px, py), 5, (255, 255, 0), -1)

    out.write(frame)
    cv2.imshow("Smart Traffic System", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
