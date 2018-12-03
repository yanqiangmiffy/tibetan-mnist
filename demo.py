# -*- coding: utf-8 -*-
# @Time    : 2018/12/3 22:02
# @Author  : quincyqiang
# @File    : demo.py
# @Software: PyCharm

import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

data = np.load('input/TibetanMNIST.npz')
X, y = data['image'], data['label']  # (17768, 28, 28)
# print(X[0])
# plt.imshow(X[0],cmap='gray')
# plt.show()

X = X.reshape(17768, 784).astype('float32') / 255
y = np_utils.to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = Sequential()
model.add(Dense(input_shape=(784,), units=512, activation='relu',name="Dense1"))
model.add(Dropout(0.2,name='Dropout1'))
model.add(Dense(input_shape=(512,), units=256, activation='relu',name='Dense2'))
model.add(Dropout(0.2,name='Dropout2'))
model.add(Dense(input_shape=(256,), units=10, activation='softmax',name='Dense3'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
from keras.utils import plot_model
plot_model(model, to_file='assets/model.png',show_shapes=True)

model.fit(x=X_train, y=y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(X_test, y_test))

evaluation = model.evaluate(X_test, y_test, verbose=1)

print("loss:%.2f,accuracy:%.2f" % (evaluation[0], evaluation[1]))
