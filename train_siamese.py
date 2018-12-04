# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: train_siamese.py 
@Time: 2018/12/4 10:42
@Software: PyCharm 
@Description:
"""
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten
from keras.layers import Dropout, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from utils import *

# 准备数据
data = np.load('input/TibetanMNIST.npz')
X, y = data['image'], data['label']  # (17768, 28, 28)
X = X.astype('float32') / 255
# y = np_utils.to_categorical(y, num_classes=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# 创建正负样本 比如 x为[img_0,img_0],那么label为1，x为[img_0,img_1],label为0
def create_pairs(x, digit_indices, num_classes):
    pairs = []
    labels = []
    n = min([len(digit_indices[j]) for j in range(num_classes)]) - 1
    for j in range(num_classes):
        for i in range(n):
            p1, p2 = digit_indices[j][i], digit_indices[j][i + 1]
            pairs += [[x[p1], x[p2]]]
            inc = random.randrange(1, num_classes)
            jn = (j + inc) % num_classes
            p1, p2 = digit_indices[j][i], digit_indices[jn][i]
            pairs += [[x[p1], x[p2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


num_classes = len(np.unique(y_train))
# 所有数据作为训练集
digit_indices = [np.where(y_train == i)[0] for i in range(0, 10)]
train_pairs, train_y = create_pairs(X_train, digit_indices, num_classes)

digit_indices = [np.where(y_test == i)[0] for i in range(0, 10)]
test_pairs, test_y = create_pairs(X_test, digit_indices, num_classes)


# 损失函数
def contrastive_loss(y_true, y_pred):
    margin = 1
    sq_pred = K.square(y_pred)
    margin_sq = K.square(K.maximum(margin - y_pred, 0))
    loss = K.mean(y_true * sq_pred + (1 - y_true) * margin_sq)
    return loss


# 欧式距离
def euclidean_distance(vects):
    x, y = vects
    sum_sq = K.sum(K.square(x - y), axis=1, keepdims=True)
    distance = K.sqrt(K.maximum(sum_sq, K.epsilon()))
    return distance


def eucl_shape(shape):
    shape1, shape2 = shape
    return (shape1[0], 1)


# 创建Siamese模型
def shared_network(input_shape):
    """
    共享层
    :param input_shape:
    :return:
    """
    input = Input(shape=input_shape)
    layer = Flatten()(input)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(128, activation='relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(128, activation='relu')(layer)
    return Model(input, layer)


input_shape = X_train.shape[1:]
siamese = shared_network(input_shape)
input_left = Input(shape=input_shape)
input_right = Input(shape=input_shape)
output_left = siamese(input_left)
output_right = siamese(input_right)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_shape)([output_left, output_right])

# ------------------------------------------------------------------------------
#   创建模型并训练
# ------------------------------------------------------------------------------
model = Model([input_left, input_right], distance)
ada = Adadelta()
threshold = 0.5


def acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


model.compile(loss=contrastive_loss, optimizer=ada, metrics=[acc])
filepath = "model/model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='atuo')

history = model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_y,
                    batch_size=128,
                    epochs=20,
                    callbacks=[checkpoint],
                    validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_y))

# 加载模型 best for test
model.load_weights("model/model.hdf5")


# 评估模型
def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)


# ------------------------------------------------------------------------------
y_pred = model.predict([train_pairs[:, 0], train_pairs[:, 1]])
train_acc = compute_accuracy(train_y, y_pred)
y_pred = model.predict([test_pairs[:, 0], test_pairs[:, 1]])
test_acc = compute_accuracy(test_y, y_pred)
print(' Accuracy on training set:', train_acc)
print('Accuracy on test set:', test_acc)
# draw_train(history)


x1 = train_pairs[:, 0][3072]
x2 = train_pairs[:, 1][0]
draw_img(x1)
draw_img(x2)

x1 = x1.reshape(1, 28, 28)
x2 = x2.reshape(1, 28, 28)
y_pred = model.predict([x1, x2])
print(y_pred)
# print(y_pred.ravel())
pred = y_pred.ravel() < threshold
print(pred)

print(np.mean(pred == 1))
