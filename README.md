# tibetan
https://www.kesci.com/home/dataset/5bfe734a954d6e0010683839
## 模型测试

- 3层全连接网络，两层Dropout

![](https://github.com/yanqiangmiffy/tibetan/blob/master/assets/model_mlp.png)

结果如下，在测试集准确率达到99.46%

> loss: 0.0029 - acc: 0.9989 - val_loss: 0.0216 - val_acc: 0.9946

- CNN

可以达到98%的准确率
```text
loss: 0.0269 - acc: 0.9910 - val_loss: 0.0560 - val_acc: 0.9829
```

预测正确的情况：
![](https://github.com/yanqiangmiffy/tibetan/blob/master/assets/cnn_right.png)

预测错误的情况：
![](https://github.com/yanqiangmiffy/tibetan/blob/master/assets/cnn_false.png)


reference:https://www.one-tab.com/page/OeLJRrcFTW2DthBklKUMZw
