# VGG_network_learn
学习搭建VGG卷积神经网络成功后记录一下

主要环境：numpy==1.21.6 keras==2.10.0 tensorflow==2.10.0 opencv-python==4.6.0.66 Pillow==9.2.0 scikit-image matplotlib==3.5.3

face_get.py是使用cv2借助摄像头截取人脸照片并保存，需要根目录下存在haarcascade_frontalface_default.xml文件（这是cv2官方的人脸识别模型）

data_strong.py是对原始图片进行旋转、添加噪声以实现数据增强（我使用是只截取了26张照片，用来训练肯定是不够的，于是我增加了数据增强）

data_separate.py是用来划分数据集的，注意里面的目录是不能重复出现的，不然会报错

network.py是我个人搭建VGG神经网络以及选取优化器

params.py是神经网络、训练模型、测试模型的各种参数

train.py是模型训练，训练好的模型将会保存在model文件夹中

detect.py是对模型进行测试
