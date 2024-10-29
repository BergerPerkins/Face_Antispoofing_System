import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json


#face_cascade = cv2.CascadeClassifier(r"D:\Deep Flow Technologies\face_spoofing_detection\gpt\haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

json_file = open("antispoofing_model_mobilenet.json",'r')# Load Anti-Spoofing Model graph
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("antispoofing_model.keras")# load antispoofing model weights 
print("Model loaded from disk")

def detect_spoofing(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No face detected")
        return

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = image[y-5:y+h+5, x-5:x+w+5]

        # Resize face to 160x160
        resized_face = cv2.resize(face, (160, 160))

        # Normalize pixel values
        resized_face = resized_face.astype("float") / 255.0

        # Expand dimensions for model input
        resized_face = np.expand_dims(resized_face, axis=0)

        # Predict spoofing
        preds = model.predict(resized_face)[0][0]

        print(f"Confidence: {preds:.2f}")

        if preds > 0.80:
            print("Spoof")
        else:
            print("Real")


# Example usage
image_path = r"D:\Deep Flow Technologies\face_spoofing_detection\Face_Antispoofing_System\final_antispoofing\test\spoof\tanka\tanka_spoof_2_face.jpg"
detect_spoofing(image_path)