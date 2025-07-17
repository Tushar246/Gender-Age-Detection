#pyright:reportMissingImports = false
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("gender_model.h5")

img = cv2.imread("test_image.jpg")
img = cv2.resize(img , (200,200))
img = img/255.0
img = np.expand_dims(img , axis=0)

prediction = model.predict(img)
class_idx = np.argmax(prediction)

labels = ['Male' , 'Female']
print("Predicted gender:" , labels[class_idx])