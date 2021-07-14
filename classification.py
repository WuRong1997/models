import numpy as np
import keras
from keras import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, BatchNormalization

## 二分类模型使用loss=keras.losses.categorical_crossentropy，即是=[1,0]，否=[0,1]
# 读取数据
x_train #(5380, 128)
x_val
x_test
y_train #(5380, 2)
y_val
# 设置model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(128,)))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=100, epochs=45, verbose=1, validation_data=(x_val))
predict_test = model.predict(x_test)

## 二分类模型使用loss='binary_crossentropy'，即是=1，否=0
# 生成虚拟数据
x_train = np.random.random((1000, 20))    #(1000, 20)
y_train = np.random.randint(2, size=(1000, 1))    #(1000, 1)
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))
# 设置model
model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=128)
score = model.evaluate(x_test, y_test, batch_size=128)

## 加载和保存模型
filepath = "F:\Study\Speech Enhancement\SE-DNN-NMF\model_save\mymodel.hdf5"
model = keras.models.load_model(filepath)
model.fit(X_train, y_train, batch_size=1024 , epochs=10)
# 保存最后一个模型及结构
model.save("F:\Study\Speech Enhancement\SE-DNN-NMF\model_save\mymodel.h5")
