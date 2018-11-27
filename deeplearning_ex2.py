import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Conv2D,Flatten
from keras.models import Sequential

RANDOM_SEED=159

X, y = fetch_openml(name='mnist_784', return_X_y = True, cache = False)
#Xdf=pd.Dataframe(X)
#ydf=pd.Dataframe(y)

#70000,784
print(X.shape)

X = X.reshape((70000, 28, 28, 1))

#print(y)
# One Hot Encoding
yoh = np.zeros((len(y), 10))
for i in range(len(y)):
    yoh[i,int(y[i])] = 1

X_train, X_test, y_train, y_test = train_test_split(X, yoh)

#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)


# Model Building
model = Sequential()
model.add(Conv2D(20, activation='relu', kernel_size=3, input_shape=(28, 28, 1), padding='same'))
model.add(Conv2D(10, activation='relu', kernel_size=3, input_shape=(28, 28, 1), padding='same'))
model.add(Flatten())
#model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.2, epochs=10)

print(model.evaluate(X_test, y_test))
