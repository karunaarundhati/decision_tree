import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
df = pd.read_csv("NBD.csv")
x = df.drop('diabetes', axis=1).astype('float32')
y = df['diabetes']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Build model
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate
loss, score2 = model.evaluate(x_test, y_test)
print(f"DNN ACCURACY: {score2:.4f}")

# Predictions
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype("int32")
print("Sklearn Accuracy:", accuracy_score(y_test, y_pred_classes))

X_tests=np.array([[45,63]],dtype=float)
print(model.predict(X_tests))