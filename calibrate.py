import cv2
import numpy as np
import sys
import os

# Ensure the data directory and video exist
VIDEO_PATH = 'data/store_feed.mp4'  # Update this if your video is named differently

if not os.path.exists(VIDEO_PATH):
    print(f"Error: Could not find the video at '{VIDEO_PATH}'.")
    print("Please ensure you downloaded it and placed it in the 'data/' folder.")
    sys.exit(1)

# Global list to store the clicked pixel coordinates
points = []

def select_points(event, x, y, flags, param):
    """Callback function to capture mouse clicks on the video frame."""
    global points
    
    # Listen for the Left Mouse Button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Prevent selecting more than 4 points
        if len(points) < 4:
            points.append([x, y])
            print(f"Point {len(points)} registered at: ({x}, {y})")
            
            # Draw a highly visible neon green dot at the clicked location
            cv2.circle(first_frame, (x, y), 6, (0, 255, 0), -1)
            
            # Draw a line connecting the points to help visualize the shape
            if len(points) > 1:
                cv2.line(first_frame, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            # Close the shape on the 4th click
            if len(points) == 4:
                cv2.line(first_frame, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
                
            cv2.imshow("Calibration: Click 4 points (Press 'q' to exit)", first_frame)

# 1. Load the video and extract only the first frame
cap = cv2.VideoCapture(VIDEO_PATH)
ret, first_frame = cap.read()
cap.release() # Immediately release memory, we only need the image

if not ret:
    print("Error: Could not read the video file. It might be corrupted.")
    sys.exit(1)

# 2. Setup the OpenCV GUI window and attach the mouse listener
window_name = "Calibration: Click 4 points (Press 'q' to exit)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_points)

print("\n" + "="*40)
print("       CALIBRATION INSTRUCTIONS")
print("="*40)
print("1. A window will open showing the first frame of the video.")
print("2. Click exactly 4 points on the floor.")
print("3. These points MUST form a perfect rectangle in the real physical world.")
print("4. Try to select a wide, spread-out area to improve calculation accuracy.")
print("5. The window will close automatically after the 4th click.")
print("   (Or press 'q' at any time to quit early)\n")

# 3. Display the frame and wait for user interaction
cv2.imshow(window_name, first_frame)

while True:
    # Wait for 1ms and check for keypresses
    key = cv2.waitKey(1) & 0xFF
    
    # Break the loop if 4 points are collected or 'q' is pressed
    if len(points) == 4:
        print("\nAll 4 points collected.")
        # Pause for a brief second so you can see the closed green shape
        cv2.waitKey(1000) 
        break
    elif key == ord('q'):
        print("\nCalibration cancelled by user.")
        break

# 4. Cleanly destroy the OpenCV windows (Includes M1 Mac fix)
cv2.destroyAllWindows()
cv2.waitKey(1) 

# 5. Output the final array format needed for the main pipeline
print("\n" + "="*40)
print("          CALIBRATION COMPLETE")
print("="*40)

if len(points) == 4:
    print("\nSUCCESS! Copy the following line of code into your main.py script:\n")
    print(f"SOURCE_POINTS = np.array({points}, dtype='float32')\n")
else:
    print(f"\nWarning: You only selected {len(points)} points.")
    print("You need exactly 4 points to calculate a Homography matrix. Run the script again.")