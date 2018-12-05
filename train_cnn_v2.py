# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: train_cnn.py 
@Time: 2018/12/4 14:09
@Software: PyCharm 
@Description:
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.normalization import  BatchNormalization
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.optimizers import Adadelta
import keras.backend as K
import matplotlib.pyplot as plt
# 参数设置
batch_size=128
n_classes=10
n_epochs=20

img_rows=28
img_cols=28

# 加载数据
data = np.load('input/TibetanMNIST.npz')
X, y = data['image'], data['label']  # (17768, 28, 28)
X = X.reshape(X.shape[0], img_rows,img_cols,1).astype('float32') / 255
X_train, X_test, y_train, y_test_ = train_test_split(X, y, random_state=42)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test_, num_classes=10)

# 构建模型
input_shape=(img_rows,img_cols,1)
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


predicted_classes = model.predict_classes(X_test)
correct_indices = np.nonzero(predicted_classes == y_test_)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test_)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test_[correct]))
plt.show()

plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test_[incorrect]))
plt.show()