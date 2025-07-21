#pyright: reportMissingImports = false
# predict_age_gender.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Constants
AGE_MODEL_PATH = 'age_model.h5'
GENDER_MODEL_PATH = 'gender_model.h5'
IMAGE_SIZE = 200
GENDER_LABELS = ['Female', 'Male']

# Load models
age_model = load_model(AGE_MODEL_PATH  , compile= False)
gender_model = load_model(GENDER_MODEL_PATH)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or could not be read.")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 200, 200, 3)
    return img

def predict(image_path):
    img = preprocess_image(image_path)

    # Predict age
    predicted_age = int(age_model.predict(img)[0][0])

    # Predict gender
    gender_pred = gender_model.predict(img)[0][0]
    predicted_gender = GENDER_LABELS[int(round(gender_pred))]

    return predicted_age, predicted_gender

# Example usage
if __name__ == '__main__':
    path = 'test_image3.jpg'  # Replace with any image path
    age, gender = predict(path)
    print(f"Predicted Age: {age}")
    print(f"Predicted Gender: {gender}")
