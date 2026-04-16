import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Фирменные цвета
COLOR_NEON_GREEN = (0, 215, 27)  # #1bd700
COLOR_PURPLE = (145, 13, 110)    # #6e0d91
COLOR_BG = (17, 24, 17)          # #111811

cap = cv2.VideoCapture(0)
success, first_frame = cap.read()
if not success:
    print("Не удалось запустить камеру!")
    exit()
h, w, c = first_frame.shape

def load_and_prep_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        dummy = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
        cv2.putText(dummy, f"Not found: {os.path.basename(path)}", (20, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NEON_GREEN, 2)
        return dummy
        
    img_resized = cv2.resize(img, (w, h))
    
    if img_resized.shape[2] == 3:
        return img_resized
        
    alpha_channel = img_resized[:, :, 3] / 255.0
    color_channels = img_resized[:, :, :3]
    background = np.full((h, w, 3), COLOR_BG, dtype=np.uint8)
    
    for c_idx in range(3):
        background[:, :, c_idx] = (alpha_channel * color_channels[:, :, c_idx] +
                                  (1 - alpha_channel) * background[:, :, c_idx])
    return background


# --- НАСТРОЙКА КАРТИНОК ЭМОЦИЙ ---
images = {
    "neutral": load_and_prep_img("assets/frog_neutral.png"),
    "happy": load_and_prep_img("assets/frog_happy.png"), 
    "sad": load_and_prep_img("assets/frog_sad.png"), 
    "pointing": load_and_prep_img("assets/frog_pointing.png"), 
    "peace": load_and_prep_img("assets/frog_peace.png") 
}

frame_counter = 0
current_emotion = "neutral"
current_state = "neutral"

# --- НАСТРОЙКА ЧУВСТВИТЕЛЬНОСТИ ЭМОЦИЙ ---
EMOTION_THRESHOLD = 20.0 

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    
    # 1. Распознавание рук
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    gesture = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            fingers = []
            
            for tip, pip in zip(tips, pips):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
            if fingers == [1, 0, 0, 0]:
                gesture = "pointing"
            elif fingers == [1, 1, 0, 0]:
                gesture = "peace"
            else:
                gesture = "pointing"
            break

    # 2. Распознавание эмоций
    if frame_counter % 15 == 0:
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Достаем словарь с процентами по каждой эмоции
            emotions_dict = analysis['emotion']
            
            # Кастомная логика чувствительности
            if emotions_dict.get('happy', 0) > EMOTION_THRESHOLD:
                current_emotion = "happy"
            elif emotions_dict.get('sad', 0) > EMOTION_THRESHOLD:
                current_emotion = "sad"
            else:
                current_emotion = "neutral"
                
        except Exception:
            pass 

    frame_counter += 1

    # 3. Приоритеты состояний
    if gesture:
        current_state = gesture
    elif current_emotion == "happy":
        current_state = "happy"
    elif current_emotion == "sad":
        current_state = "sad"
    else:
        current_state = "neutral"

    frog_img = images[current_state]

    # 4. Сборка интерфейса
    separator = np.full((h, 10, 3), COLOR_PURPLE, dtype=np.uint8)
    
    combined_window = np.hstack((frame, separator, frog_img))

    cv2.putText(combined_window, f"Emotion: {current_emotion}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NEON_GREEN, 2, cv2.LINE_AA)
    cv2.putText(combined_window, f"State: {current_state}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NEON_GREEN, 2, cv2.LINE_AA)

    cv2.imshow("From Zero To Frog", combined_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()