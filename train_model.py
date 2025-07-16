# pyright: reportMissingImports=false

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from preprocess import load_data

X , y_age , y_gender = load_data(limit = 5000)

X = X/255.0

y_gender_cat = to_categorical(y_gender , 2)

X_train, X_test, y_train, y_test = train_test_split(X, y_gender_cat, test_size=0.2, random_state=42)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Flatten , Dense , Dropout

model = Sequential([
    Conv2D(32 , (3,3) , activation ='relu' , input_shape=(200,200,3)),
    MaxPooling2D(2,2),

    Conv2D(64 , (3,3) , activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128 , activation = 'relu'),
    Dropout(0.3),
    Dense(2 , activation='softmax')
])

model.compile(optimizer ='adam' , loss='categorical_crossentropy',metrics = ['accuracy'])

model.fit(X_train , y_train , epochs =10 , batch_size=32 , validation_split=0.2)

model.evaluate(X_test , y_test)

model.save("gender_model.h5")