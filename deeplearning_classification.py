# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Keras specific
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical 
X = np.array([[30],[40],[50],[60],[20],[10],[70]],dtype=float)
y = np.array([0,1,1,1,0,0,1],dtype=float)
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=1))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model.fit(X,y, epochs=100)
X_marks=np.array([[20]],dtype=float)
print(model.predict(X_marks))
