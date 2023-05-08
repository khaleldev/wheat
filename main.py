import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture('wheat1.mp4')

# Define the lower and upper bounds of the green color in HSV color space
lower_green = np.array([40, 40, 40])
upper_green = np.array([70, 255, 255])

# Loop through each frame in the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the image to get the green regions
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours of the green regions
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and draw a yellow rectangle around it
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"{w}x{h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow('frame', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()