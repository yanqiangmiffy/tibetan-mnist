# # input_shape=(28,28,1)
# model=Sequential()
# model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=input_shape,activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(BatchNormalization())
#
# model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
#
# model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(n_classes, activation='softmax'))
#
# model.summary()
