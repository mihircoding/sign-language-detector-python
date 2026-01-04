import os
import pickle

import mediapipe as mp
import cv2
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task', delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.3
)

DATA_DIR = './data'

data = []
labels = []

with HandLandmarker.create_from_options(options) as landmarker:
    for dir_ in os.listdir(DATA_DIR):
        for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    for landmark in hand_landmarks:
                        x = landmark.x
                        y = landmark.y

                        x_.append(x)
                        y_.append(y)

                    for landmark in hand_landmarks:
                        x = landmark.x
                        y = landmark.y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()