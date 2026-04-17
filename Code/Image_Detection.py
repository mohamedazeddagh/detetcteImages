# Image_Detection.py
import cv2
import numpy as np

def sift_detector(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    image2 = image_template

    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=3)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return len(good_matches)


image_template = [cv2.imread('Images/Test_Run_Images/Face_Testing.jpg', 0)]  # Reference image

def live_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        top_left_x = int(width / 3)
        top_left_y = int(height / 4)
        bottom_right_x = int((width / 3) * 2)
        bottom_right_y = int((height / 2))

        # Draw ROI rectangle
        cv2.rectangle(frame, (top_left_x, bottom_right_y), (bottom_right_x, top_left_y), (255, 0, 0), 3)
        cropped = frame[bottom_right_y:top_left_y, top_left_x:bottom_right_x]

        # Flip frame horizontally
        frame_flip = cv2.flip(frame, 1)

        matches = sift_detector(cropped, image_template[0])
        cv2.putText(frame, f'Matches: {matches}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if matches > 10:
            cv2.putText(frame, 'Object Found', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Live Feed', frame)
        if cv2.waitKey(1) == 27:  # Esc key
            break

    cap.release()
    cv2.destroyAllWindows()