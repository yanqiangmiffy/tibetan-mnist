# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: train_lr.py
@Time: 2018/12/4 10:28
@Software: PyCharm 
@Description:
"""
import numpy as np
from keras.models import  Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

data = np.load('input/TibetanMNIST.npz')
X, y = data['image'], data['label']  # (17768, 28, 28)
# print(X[0])
# plt.imshow(X[0],cmap='gray')
# plt.show()

X = X.reshape(17768, 784).astype('float32') / 255
y = np_utils.to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model=Sequential()
model.add(Dense(input_shape=(784,),units=10,activation='softmax'))
model.compile(optimizer=RMSprop(lr=0.001,rho=0.9),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.save('data/regression.hdf5')
history=model.fit(X_train,y_train,
                  batch_size=128,
                  epochs=20,
                  verbose=1,
                  validation_data=(X_test,y_test))
evaluation=model.evaluate(X_test,y_test)
print('Summary: Loss over the test dataset: %.4f, Accuracy: %.4f' % (evaluation[0], evaluation[1]))


