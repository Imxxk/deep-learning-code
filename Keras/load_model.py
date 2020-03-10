# -*- coding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda,Dense,Dropout,Activation,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
#加载训练数据
(X_train,y_train),(X_test,y_test)=mnist.load_data()

#训练数据预处理
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)

#构建神经网络
# def build_il_model():
#     model = Sequential()
#     model.add(Conv2D(24, (5, 5), activation='elu',strides=2,input_shape=(28,28,1)))
#     model.add(Conv2D(36, (5, 5), activation='elu',strides=2))
#     model.add(Conv2D(48, (2, 2), activation='elu'))
#     model.add(Conv2D(64, (3, 3), activation='elu',border_mode='same'))
#     model.add(Conv2D(64, (3, 3), activation='elu',border_mode='same'))
#     # model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(1000, activation='elu'))
#     model.add(Dense(500, activation='elu'))
#     model.add(Dense(10, activation='softmax'))
#     model.summary()
#     return model

#模型训练
# def train_model(model):
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=1, batch_size=32)

#开始训练
def main():
    modelone=model_from_json(open(r'C:\Users\xiongkang\PycharmProjects\gpu\keras\model_architecture.json').read())
    modelone.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    # load_model(modelone)
    modelone=load_model('mymodel.h5')
    print('\nTesting------------------')
    loss,accuracy= modelone.evaluate(X_test, y_test)
    print('\ntest loss:', loss)
    print('\ntest accuracy:', accuracy)


if __name__ == '__main__':
    main()
