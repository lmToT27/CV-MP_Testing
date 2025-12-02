import cv2, mediapipe as mp
from SaoMeoEngine import SaoMeoEngine

# Initialize SaoMeo sound engine and define musical notes
my_sao_meo = SaoMeoEngine()

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

mp_hands = mp.solutions.hands
mp_hands_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode = False,
                       max_num_hands = 2,
                       min_detection_confidence = 0.7,
                       min_tracking_confidence = 0.3)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Opening camera... Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        current_notes = []
        current_hands = {}

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                current_hands[label] = hand_landmarks
                mp_hands_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if "Left" in current_hands:
            pass
        if "Right" in current_hands:
            pass
            
        cv2.imshow("Play Sao Meo with hands", img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    my_sao_meo.close()
    cap.release()
    cv2.destroyAllWindows()