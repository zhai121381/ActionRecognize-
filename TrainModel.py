# coding: utf-8
import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from tensorflow import keras
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
#from keras import optimizers
from keras.models import load_model



class TrainNetWork():
    def Train(self):
        # 动作数据存放文件夹
        Action_File = "ActionData"

        # 动作模型
        MODEL_NAME = "ActionModel.h5"

        # 动作标签
        LABEL_NAME = "ActionLabels.dat"

        # 初始化
        data = []
        labels = []
        class_num = 0
        train_num = 20

        class_num = int(input('请输入需要进行学习的动作个数：'))
        train_num = int(input('请输入训练迭代次数：'))

        #遍历动作数据文件夹
        for image_file in paths.list_images(Action_File):

            #从动作数据文件夹中读取数据
            image = cv2.imread(image_file)

            # 增加维度
            image = cv2.resize(image, (100,100))

            #获取动作与其相对应的名称
            label = image_file.split(os.path.sep)[-2]

            #将标签和数据添加进data和label
            data.append(image)
            labels.append(label)

        print('数据标签加载完成')

        #归一化
        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        #测试集，数据集分离
        (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.2, random_state=0)

        # 标签二值化
        lb = LabelBinarizer().fit(Y_train)
        Y_train = lb.transform(Y_train)
        Y_test = lb.transform(Y_test)


        #标签数据文件保存
        with open(LABEL_NAME, "wb") as f:
            pickle.dump(lb, f)
        print('生成dat文件，开始构建神经网络')

        #神经网络构建
        model = Sequential()

        # 第一层卷积池化
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=(100,100, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 第二层卷积池化
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


        # 第三层卷积池化
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


        # 全连接层
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))

        #防止过拟合
        model.add(Dropout(0.5))

        #输出层(用的是多分类的对数损失函数softmax)
        model.add(Dense(class_num, activation="softmax"))
        model.summary()

        #建立优化器(loss用的是categorical_crossentropy

        model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.RMSprop(lr=0.0001), metrics=["accuracy"])
        print('构建成功，开始训练')

        print('模型导入完成')
        history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), batch_size=32, epochs=train_num, verbose=1)

        #保存模型
        model.save(MODEL_NAME)

        print('训练保存完成')

