import cv2
import mediapipe as mp
import numpy as np
import time
import threading
from gtts import gTTS
import pygame
from PIL import Image, ImageDraw, ImageFont
import os

# ================= INITIALIZATION ================= #

pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=512)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2   # 🔥 TWO HANDS
)
mp_draw = mp.solutions.drawing_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

font_en = ImageFont.truetype(
    os.path.join(BASE_DIR, "NotoSansDevanagari-VariableFont_wdth,wght.ttf"), 36
)
font_kn = ImageFont.truetype(
    os.path.join(BASE_DIR, "NotoSansKannada-VariableFont_wdth,wght.ttf"), 36
)
font_hi = ImageFont.truetype(
    os.path.join(BASE_DIR, "NotoSansDevanagari-VariableFont_wdth,wght.ttf"), 36
)

AUDIO_FILE = "speech_temp.mp3"
SPEAKING = False

MODE = "NUMBER"
DISPLAY_WORDS = None
LAST_SPOKEN = None

# ================= NUMBER WORDS ================= #

NUMBER_WORDS = {
    0: {'en': 'Zero', 'kn': 'ಸೊನ್ನೆ', 'hi': 'शून्य'},
    1: {'en': 'One', 'kn': 'ಒಂದು', 'hi': 'एक'},
    2: {'en': 'Two', 'kn': 'ಎರಡು', 'hi': 'दो'},
    3: {'en': 'Three', 'kn': 'ಮೂರು', 'hi': 'तीन'},
    4: {'en': 'Four', 'kn': 'ನಾಲ್ಕು', 'hi': 'चार'},
    5: {'en': 'Five', 'kn': 'ಐದು', 'hi': 'पांच'},
    6: {'en': 'Six', 'kn': 'ಆರು', 'hi': 'छह'},
    7: {'en': 'Seven', 'kn': 'ಏಳು', 'hi': 'सात'},
    8: {'en': 'Eight', 'kn': 'ಎಂಟು', 'hi': 'आठ'},
    9: {'en': 'Nine', 'kn': 'ಒಂಬತ್ತು', 'hi': 'नौ'},
    10:{'en': 'Ten', 'kn': 'ಹತ್ತು', 'hi': 'दस'}
}

# ================= FINGER COUNT ================= #

def count_fingers(hand_landmarks):
    lm = hand_landmarks.landmark
    count = 0

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for t, p in zip(tips, pips):
        if lm[t].y < lm[p].y:
            count += 1

    thumb = np.array([lm[4].x, lm[4].y])
    index = np.array([lm[5].x, lm[5].y])
    wrist = np.array([lm[0].x, lm[0].y])

    if np.linalg.norm(thumb - index) > np.linalg.norm(index - wrist) * 0.6:
        count += 1

    return count

# ================= SPEECH ================= #

def speak(text, lang='en'):
    global SPEAKING
    SPEAKING = True
    tts = gTTS(text=text, lang=lang)
    tts.save(AUDIO_FILE)
    pygame.mixer.music.load(AUDIO_FILE)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    os.remove(AUDIO_FILE)
    SPEAKING = False

def speak_multilang(words):
    speak(words['en'], 'en')
    speak(words['kn'], 'kn')
    speak(words['hi'], 'hi')

# ================= STATIC ALPHABET ================= #

def detect_static_alphabet(lm):
    fingers = []

    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for t, p in zip(tips, pips):
        fingers.append(lm[t].y < lm[p].y)

    thumb_open = lm[4].x < lm[3].x
    fingers.insert(0, thumb_open)

    pattern = tuple(fingers)

    ALPHABET_MAP = {
        (False, False, False, False, False): 'A',
        (False, True, True, True, True): 'B',
        (False, True, False, False, False): 'D',
        (True, True, False, False, False): 'L',
        (False, True, True, False, False): 'V',
        (False, True, True, True, False): 'W',
        (True, False, False, False, True): 'Y'
    }

    return ALPHABET_MAP.get(pattern, None)

# ================= CAMERA ================= #

cap = cv2.VideoCapture(0)

# ================= MAIN LOOP ================= #

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    detected_letter = None

    if result.multi_hand_landmarks:
        for hand_lm in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

        if MODE == "NUMBER":
            total = 0
            for hand_lm in result.multi_hand_landmarks:
                total += count_fingers(hand_lm)

            if total != LAST_SPOKEN and total in NUMBER_WORDS and not SPEAKING:
                DISPLAY_WORDS = NUMBER_WORDS[total]
                threading.Thread(
                    target=speak_multilang,
                    args=(DISPLAY_WORDS,),
                    daemon=True
                ).start()
                LAST_SPOKEN = total

        else:
            detected_letter = detect_static_alphabet(result.multi_hand_landmarks[0].landmark)
            if detected_letter and detected_letter != LAST_SPOKEN and not SPEAKING:
                threading.Thread(
                    target=speak,
                    args=(detected_letter,),
                    daemon=True
                ).start()
                LAST_SPOKEN = detected_letter

    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text((20,20), f"MODE: {MODE}", font=font_en, fill=(255,255,0))

    if MODE == "NUMBER" and DISPLAY_WORDS:
        draw.text((20,70), DISPLAY_WORDS['en'], font=font_en, fill=(0,255,0))
        draw.text((20,120), DISPLAY_WORDS['kn'], font=font_kn, fill=(0,255,0))
        draw.text((20,170), DISPLAY_WORDS['hi'], font=font_hi, fill=(0,255,0))

    if MODE == "ALPHABET" and detected_letter:
        draw.text((20,70), f"Letter: {detected_letter}", font=font_en, fill=(0,255,0))

    frame = np.array(frame_pil)
    cv2.imshow("Finger & Alphabet Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('a'):
        MODE = "ALPHABET"
        LAST_SPOKEN = None
    elif key == ord('n'):
        MODE = "NUMBER"
        LAST_SPOKEN = None
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
