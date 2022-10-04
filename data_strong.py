import cv2, os, random
import numpy as np
from skimage import util, img_as_float, io
from PIL import Image


def rand_list(num, start=1, end=10):
    munber = []
    while num > 0:
        num -= 1
        munber.append(random.randint(start, end))
    return munber


def Rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated


def add_noise_Guass(img, mean=0, var=0.02):  # 添加高斯噪声
    img = np.array(img/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out_img = img + noise
    if out_img.min() < 0:
        low_clip = -1
    else:
        low_clip = 0
        out_img = np.clip(out_img, low_clip, 1.0)
        out_img = np.uint8(out_img * 255)
    return out_img


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
# print(rand_list(10, 1, 25))
start_num = 25
root_pth = "./images/"
dir_pic, img_name = get_pic_pth(root_pth)
print(dir_pic, img_name)
for t in range(4):
    last_dir = ''
    rand_mun = rand_list(20, 1, len(img_name)//2)
    for i in dir_pic[0]:
        if last_dir != i:
            last_dir = i
            num = 0
        for j in rand_mun:
            pic_pth = root_pth+i+'/'+i+str(j)+'.jpg'
            img = cv2.imread(pic_pth)
            img = Rotate(img, random.randint(0, 180))
            num += 1
            cv2.imwrite(root_pth+i+'/'+i+str(start_num+num)+'.jpg', img)
    start_num = start_num + num
dir_pic, img_name = get_pic_pth(root_pth)
rand_mun = rand_list(40, 1, len(img_name)//2)
for i in dir_pic[0]:
    if last_dir != i:
        last_dir = i
        num = 0
    for j in rand_mun:
        pic_pth = root_pth + i + '/' + i + str(j) + '.jpg'
        img = cv2.imread(pic_pth)
        img = add_noise_Guass(img) * 255
        num += 1
        cv2.imwrite(root_pth + i + '/' + i + str(start_num + num) + '.jpg', img)
start_num = start_num + num
rand_mun = rand_list(40, 1, len(img_name)//2)
for i in dir_pic[0]:
    if last_dir != i:
        last_dir = i
        num = 0
    for j in rand_mun:
        pic_pth = root_pth + i + '/' + i + str(j) + '.jpg'
        img = Image.open(pic_pth)
        img = img_as_float(img)
        img = util.random_noise(img, mode='s&p')
        num += 1
        io.imsave(root_pth + i + '/' + i + str(start_num + num) + '.jpg', img)
start_num = start_num + num

cv2.waitKey(0)



