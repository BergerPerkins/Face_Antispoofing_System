import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tqdm import tqdm
import time

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

json_file = open('antispoofing_model_mobilenet.json','r')# Load Anti-Spoofing Model graph
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('antispoofing_model.keras')# load antispoofing model weights 
print("Model loaded from disk")

for img in tqdm(os.listdir(os.path.join(root_dir,'final_antispoofing/test/my'))):
    t1 = time.time()
    img_arr = cv2.imread(os.path.join(root_dir,'final_antispoofing/test/my',img))
    resized_face = cv2.resize(img_arr,(160,160))
    resized_face = resized_face.astype("float") / 255.0
    # resized_face = img_to_array(resized_face)
    resized_face = np.expand_dims(resized_face, axis=0)
    # pass the face ROI through the trained liveness detector
    # model to determine if the face is "real" or "fake"
    preds = model.predict(resized_face)[0]
    # if preds> 0.5:
    #     label = 'spoof'
    #     t2 = time.time()
    #     print( 'Time taken was {} seconds'.format( t2 - t1))
    # else:
    #     label = 'real'
    #     t2 = time.time()
    #     print( 'Time taken was {} seconds'.format( t2 - t1))
    if preds > 0.5:
        label = 'spoof'
    else:
        label = 'real'
        
    t2 = time.time()
    print(f'Image: {img}, Classification: {label}, Time taken: {t2 - t1:.2f} seconds')
