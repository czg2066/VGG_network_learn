import keras.layers as ky
import keras.models as km
import keras.optimizers as ko
import params

param = params.net_param()


class my_model():
    """自建模型"""
    def __init__(self):
        # 输入数据
        self.inputs = ky.Input(shape=(param.Height, param.Width, param.Chancel))
        # 卷积层
        self.x = ky.Conv2D(64, param.Net_kernel_size, activation='relu')(self.inputs)
        # 池化层1
        self.x = ky.MaxPooling2D(2)(self.x)
        # 卷积层
        self.x = ky.Conv2D(128, param.Net_kernel_size, activation='relu')(self.x)
        # 池化层2
        self.x = ky.MaxPooling2D(2)(self.x)
        # 卷积层
        self.x = ky.Conv2D(256, param.Net_kernel_size, activation='relu')(self.x)
        self.x = ky.Conv2D(256, param.Net_kernel_size, activation='relu')(self.x)
        # 池化层3
        self.x = ky.MaxPooling2D(2)(self.x)
        # 卷积层
        self.x = ky.Conv2D(512, param.Net_kernel_size, activation='relu')(self.x)
        self.x = ky.Conv2D(512, param.Net_kernel_size, activation='relu')(self.x)
        # 池化层4
        self.x = ky.MaxPooling2D(2)(self.x)
        # 卷积层
        self.x = ky.Conv2D(512, param.Net_kernel_size, activation='relu')(self.x)
        self.x = ky.Conv2D(512, param.Net_kernel_size, activation='relu')(self.x)
        # 池化层5
        self.x = ky.MaxPooling2D(2)(self.x)
        # 全连接
        self.x = ky.Flatten()(self.x)
        # sunshu
        self.x = ky.Dropout(0.5)(self.x)
        # FC层
        self.x = ky.Dense(4096, activation='relu')(self.x)
        self.x = ky.Dense(1024, activation='relu')(self.x)
        self.x = ky.Dense(256, activation='relu')(self.x)
        self.pre = ky.Dense(param.Classes, activation='softmax')(self.x)

        """建立模型"""
        self.model = km.Model(inputs=self.inputs, outputs=self.pre)

        """确定优化器"""
        Adam = ko.Adam(learning_rate=param.Learning_rate, epsilon=0.1)
        SGD = ko.SGD(learning_rate=param.Learning_rate, momentum=param.SGD_momentum, nesterov=param.SGD_nesterov)
        self.model.compile(optimizer=SGD, loss=param.loss, metrics=['accuracy'])

    def show(self):
        """显示网络相关参数"""
        self.model.summary()
        print("Learn_rate : ", param.Learning_rate)
        print("Loss : ", param.loss)
