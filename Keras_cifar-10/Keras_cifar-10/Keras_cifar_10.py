#show_cifar10_cnn_predict.py
#載入資料集
import time
start=time.time()
import numpy as np
np.random.seed(10)
from keras.datasets import cifar10
(x_train_image, y_train_label), (x_test_image, y_test_label)=cifar10.load_data()
import matplotlib.pyplot as plt  
#資料預處理
x_train_normalize=x_train_image.astype('float32')/255.0
x_test_normalize=x_test_image.astype('float32')/255.0
from keras.utils import np_utils
y_train_onehot=np_utils.to_categorical(y_train_label)
y_test_onehot=np_utils.to_categorical(y_test_label)

#建立模型
#建立兩層卷積 (丟棄 25% 神經元) + 池化層
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.layers import ZeroPadding2D,Activation




def show_train_history(train_history):
    fig=plt.gcf()
    fig.set_size_inches(16, 6)
    plt.subplot(121)
    plt.plot(train_history.history["acc"])
    plt.plot(train_history.history["val_acc"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "validation"], loc="upper left")
    plt.subplot(122)
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])
    plt.title("Train History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()







model=Sequential()
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 padding='same',
                 input_shape=(32,32,3),
                 activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
#建立分類模型 MLP
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.summary()

#訓練模型
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_train_normalize, y=y_train_onehot, validation_split=0.2, epochs=10, batch_size=128,verbose=2)
elapsed=time.time()-start
print("Training time=" + str(elapsed) + " Seconds")
show_train_history(train_history)
#繪製訓練結果


show_train_history(train_history)

#評估預測準確率
scores=model.evaluate(x_test_normalize, y_test_onehot)  
print("Accuracy=", scores[1])
prediction=model.predict_classes(x_test_normalize)
print(prediction)

#預測測試集圖片
prediction=model.predict_classes(x_test_normalize)
print(prediction)
