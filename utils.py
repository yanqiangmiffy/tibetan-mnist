# !/usr/bin/env python3  
# -*- coding:utf-8 _*-  
""" 
@Author:yanqiang 
@File: utils.py 
@Time: 2018/12/4 10:39
@Software: PyCharm 
@Description:
"""
import matplotlib.pyplot as plt


def draw_train(history):
    '''
    绘制训练曲线
    :param history:
    :return:
    '''

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def draw_img(X):
    """
    画图像
    :return:
    """
    plt.imshow(X, cmap='gray')
    plt.show()
