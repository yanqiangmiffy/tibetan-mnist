# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: evaluate.py 
@Time: 2018/12/7 14:53
@Software: PyCharm 
@Description:
"""
import numpy as np
from keras.models import  Sequential
from keras.models import  load_model
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
print(X_test[0].shape)
model=load_model('data/regression.hdf5')
pred=model.predict(X_test[0].reshape(1,784))
