from keras import backend as K
import tensorflow as tf
import cv2, os
import numpy as np
import network, params

param = params.net_param()
mymodel = network.my_model()
model = mymodel.model
model.load_weights(param.Model_save_pth+'four.h5')
# 预测
# 'C:\\Users\\86184\\Desktop\\k210_learn\\images\\czg\\czg180.jpg'
# 加载OpenCV人脸检测分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()  # 准备好识别方法LBPH方法


camera = cv2.VideoCapture(1)  # 0:开启摄像头
name = ['czg', 'lx']
while True:
    num_pic = 0
    ret, img = camera.read()
    face_detector = face_cascade  # 记录摄像头记录的每一帧的数据，让Classifier判断人脸
    faces = face_detector.detectMultiScale(img.copy(), 1.3, 3)  # gray是要灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
    key = cv2.waitKey(1)
    for (x, y, w, h) in faces:  # 制造一个矩形框选人脸(xy为左上角的坐标,w为宽，h为高)
        img2 = img[y:y+h, x:x+w, :].copy()
        img2 = cv2.resize(img2, (param.Width, param.Height))
        print(img2.shape)
        img_array = tf.keras.preprocessing.image.img_to_array(img2)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        preds = model.predict(img_array, steps=1)
        score = preds[0]

        print(score)
        try:
            cv2.imshow("img2", img2)
        except:
            pass
        cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
        cv2.putText(img, name[list(score).index(max(score))]+str(max(score)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        #cv2.rectangle(img, (x - 20, y - 20), (x + w + 20, y + w + 20), (0, 0, 255))
    cv2.imshow("img", img)
    #print(img.shape)

