import cv2, os, shutil
import numpy as np


def get_pic_pth(root="./images/"):
    dir = []
    img = []
    for path, dirs, files in os.walk(root):
        # print('path', path)
        # print('dirs', dirs)
        if dirs:
            dir.append(dirs)
        # print('files', files)
        for file in files:
            if '.jpg' in file:
                img.append(file)
    return dir, img


def make_train_vail():
    np.random.seed(100)
    root_pth = "./images/"
    dir_pic, img_name = get_pic_pth(root_pth)
    rand_num = []
    while True:
        i = np.random.randint(1, len(img_name)//2)
        if i in rand_num:
            continue
        rand_num.append(i)
        if len(rand_num) >= len(img_name)//2 * 0.2:
            break

    os.mkdir('./data/')
    os.mkdir('./data/train/')
    os.mkdir('./data/vali/')
    train_pth = './data/train/'
    vali_pth = './data/vali/'

    for j in dir_pic[0]:
        os.mkdir(train_pth + j)
        for i in img_name:
            if j in i:
                shutil.copy(root_pth+j+'/'+i, train_pth+j+'/'+i)
            else:
                pass
    f = open('./vali.txt', 'w')
    for j in dir_pic[0]:
        os.mkdir(vali_pth+j)
        for i in rand_num:
            shutil.move(train_pth+j+'/'+j+str(i)+'.jpg', vali_pth+j+'/'+j+str(i)+'.jpg')
            f.write(vali_pth+j+'/'+j+str(i)+'.jpg'+'\n')
    f.close()
    dir_pic, img_name = get_pic_pth(root=train_pth)

    f = open('./train.txt', 'w')
    for i in dir_pic[0]:
        for j in img_name:
            f.write(train_pth+i+'/'+j+'\n')
    f.close()


make_train_vail()