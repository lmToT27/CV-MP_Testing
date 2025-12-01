import cv2
import mediapipe as mp

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

# 1. INITIALIZE MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils # Utility to draw landmarks and connections

# Configuration: static_image_mode=False (optimized for video), max_num_hands=2
hands = mp_hands.Hands(static_image_mode = False,
                       max_num_hands = 2,
                       min_detection_confidence = 0.75,
                       min_tracking_confidence = 0.5)

# 2. INITIALIZE OPENCV
cap = cv2.VideoCapture(0) # '0' represents the default webcam

print("Opening camera... Press 'q' to exit.")

while True:
    # --- Read frame from camera (1) ---
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    # --- Preprocess the image (2) ---
    # flip img for a mirror effect
    img = cv2.flip(img, 1)
    # convert BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --- Process the image (3) ---
    results = hands.process(imgRGB)

    # --- Process the results (4) ---
    # results.multi_hand_landmarks contains coordinates of all detected hands
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # draw landmarks and connections on the original image
            mp_drawing.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
            
            #get coordinates of the thumb tip
            h, w, c = img.shape
            cx, cy = int(hand.landmark[4].x * w), int(hand.landmark[4].y * h)
            # Draw a circle at the thumb tip
            cv2.circle(img, (cx, cy), 10, hex_to_bgr("#DF7F4C"), cv2.FILLED)

    # --- Display the output (5) ---
    cv2.imshow("Hand Tracking - lmToT27", img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()