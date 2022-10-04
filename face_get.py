import cv2, time, os

# 加载OpenCV人脸检测分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()  # 准备好识别方法LBPH方法

camera = cv2.VideoCapture(1)  # 0:开启摄像头
time.sleep(2)
while True:
    name = input("输入名字：")
    num_pic = 0
    while True:
        ret, img = camera.read()
        img.resize(480, 640, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_detector = face_cascade# 记录摄像头记录的每一帧的数据，让Classifier判断人脸
        faces = face_detector.detectMultiScale(img, 1.3, 3)# gray是要灰度图像，1.3为每次图像尺寸减小的比例，5为minNeighbors
        if len(faces) >= 2:
            continue
        key = cv2.waitKey(1)
        for (x, y, w, h) in faces:  # 制造一个矩形框选人脸(xy为左上角的坐标,w为宽，h为高)
            if w*h < 480*640/40:
                continue
            if key == ord('s'):
                if num_pic == 0 and os.path.exists('./images/'+name) == False:
                    pic_path = './images/'+name
                    os.makedirs(pic_path)
                else:
                    pic_path = './images/' + name
                name_pic = pic_path + '/' + name + str(num_pic) + '.jpg'
                print(name_pic)
                num_pic += 1
                cv2.imwrite(name_pic, img[y - 20:y + h + 20, x - 20:x + w + 20])
            cv2.rectangle(img, (x, y), (x + w, y + w), (0, 255, 0))
            cv2.rectangle(img, (x-20, y-20), (x + w+20, y + w+20), (0, 0, 255))

        cv2.imshow(name, img)
        if key == ord('q') or key == ord('t'):
            break
    if key == ord('q'):
        break
