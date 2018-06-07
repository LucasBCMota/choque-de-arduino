import cv2
import numpy as np
import requests
import sys
import tensorflow as tf

import keras
from keras.preprocessing import image

class_indices = {'A': 0, 'B': 1, 'C': 2, 'Five': 3, 'Point': 4, 'V': 5}

class GestureUI():
    def __init__(self, classifier_path):
        self.classifier = self.load_classifier(classifier_path)
        self.classes = {v: k for k, v in class_indices.items()}

    def load_classifier(self, model_path):
        return keras.models.load_model(model_path)

    def classify_gesture(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (85, 92), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Input', frame)
        img_tensor = image.img_to_array(frame)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        prediction = self.classifier.predict_classes(img_tensor)[0]
        print(self.classes[prediction])

    def run(self):
        capture = cv2.VideoCapture(0)
        while True:
            _, frame = capture.read()
            cv2.imshow('CacauNet', frame)
            self.classify_gesture(frame)
            requests.post('http://127.0.0.1:5000/state', json={"mode": 1, "selection":1})
            pressed_key = cv2.waitKey(300)
            if pressed_key == 27:
                cv2.destroyAllWindows()
                sys.exit()


if __name__ == '__main__':
    ui = GestureUI('models/vanilla_conv.hdf5')
    ui.run()
