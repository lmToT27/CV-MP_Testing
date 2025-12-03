import cv2, mediapipe as mp
import math
from SaoMeoEngine import SaoMeoEngine

notes = {
    'Rest': 0,
    'C2': 65.41, 'D2': 73.42, 'E2': 82.41, 'F2': 87.31, 'G2': 98.00, 'A2': 110.00, 'B2': 123.47,
    'C2b': 61.74, 'D2b': 69.30, 'E2b': 77.78, 'F2b': 82.41, 'G2b': 92.50, 'A2b': 103.83, 'B2b': 116.54,
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
    'C3b': 123.47, 'D3b': 138.59, 'E3b': 155.56, 'F3b': 164.81, 'G3b': 185.00, 'A3b': 207.65, 'B3b': 233.08,
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
    'C4b': 246.94, 'D4b': 277.18, 'E4b': 311.13, 'F4b': 329.63, 'G4b': 369.99, 'A4b': 415.30, 'B4b': 466.16,
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
    'C5b': 493.88, 'D5b': 554.37, 'E5b': 622.25, 'F5b': 659.25, 'G5b': 739.99, 'A5b': 830.61, 'B5b': 932.33,
    'C6': 1046.50, 'D6': 1174.66, 'E6': 1318.51, 'F6': 1396.91, 'G6': 1567.98, 'A6': 1760.00, 'B6': 1975.53,
    'C6b': 987.77, 'D6b': 1108.73, 'E6b': 1244.51, 'F6b': 1318.51, 'G6b': 1479.98, 'A6b': 1661.22, 'B6b': 1864.66,
    'C7': 2093.00,
    'C7b': 1975.53
}

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
                       min_detection_confidence = 0.8,
                       min_tracking_confidence = 0.7)

def get_distance(lm1, lm2):
    point1 = (int(lm1.x * w), int(lm1.y * h))
    point2 = (int(lm2.x * w), int(lm2.y * h))
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    thumb_threshold = (1 / 10) * CAM_WIDTH
    general_threshold = (1 / 5) * CAM_WIDTH
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print("Opening camera... Press 'q' to exit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera")
            break
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        h, w, c = img.shape

        current_notes = []
        current_hands = {}

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                current_hands[label] = hand_landmarks
                mp_hands_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        increasing_half_octave = False
        flat_sound = False
        increased_octave = 0

        if "Left" in current_hands:
            hand_landmarks = current_hands["Left"]

            lm_ring_finger_mcp = hand_landmarks.landmark[13]
            lm_thumb = hand_landmarks.landmark[4]
            lm_wrist = hand_landmarks.landmark[0]

            if get_distance(lm_ring_finger_mcp, lm_thumb) > thumb_threshold:
                flat_sound = True
            
            finger_tips_ids = [8, 12, 16, 20]

            for tip_id in finger_tips_ids:
                lm = hand_landmarks.landmark[tip_id]
                if get_distance(lm, lm_wrist) > general_threshold:
                    increased_octave += 1
            
        if "Right" in current_hands:
            hand_landmarks = current_hands["Right"]

            lm_ring_finger_mcp = hand_landmarks.landmark[13]
            lm_thumb = hand_landmarks.landmark[4]
            lm_wrist = hand_landmarks.landmark[0]

            if get_distance(lm_ring_finger_mcp, lm_thumb) > thumb_threshold:
                increasing_half_octave = True

            finger_tips_ids = [8, 12, 16, 20]
            cnt = 0

            for tip_id in finger_tips_ids:
                lm = hand_landmarks.landmark[tip_id]
                if get_distance(lm, lm_wrist) > general_threshold:
                    cnt += 1
            
            note = ""
            offset = 0

            if cnt == 1:
                note = "C" if not increasing_half_octave else "G"
            elif cnt == 2:
                note = "D" if not increasing_half_octave else "A"
            elif cnt == 3:
                note = "E" if not increasing_half_octave else "B"
            elif cnt == 4:
                if increasing_half_octave: offset = 1
                note = "F" if not increasing_half_octave else "C"

            # print(increased_octave)

            if cnt: note += str(2 + offset + increased_octave) + ("b" if flat_sound else "")

            current_notes.append(note)

            cv2.putText(img, f"current notes: {current_notes}", (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

        freq_list = [notes[note] for note in current_notes if note in notes]
        my_sao_meo.update_notes(freq_list)

        cv2.imshow("Play Sao Meo with hands", img)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    
    my_sao_meo.close()
    cap.release()
    cv2.destroyAllWindows()