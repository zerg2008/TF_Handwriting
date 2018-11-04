#-*- coding: UTF-8 -*-

'''
Author: Steve Wang
Time: 2017/12/8 10:00
Environment: Python 3.6.2 |Anaconda 4.3.30 custom (64-bit) Opencv 3.3
'''
#该程序目前使用opencv可以实现如下功能：
#1、打开图片
#2、对图片进行灰度化处理
#3、对图像进行增强、细化等操作
#4、找出图像的边缘
#5、根据图像的边缘进行切割
import cv2
import numpy as np

def get_image(path):
    #获取图片
    img=cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray
#高斯去噪并转化为灰度图
def Gaussian_Blur(gray):
    # 高斯去噪
    blurred = cv2.GaussianBlur(gray, (9, 9),0)

    return blurred
#提取图像梯度并进行边缘增强
def Sobel_gradient(blurred):
    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradX, gradY, gradient
#继续去噪声
def Thresh_and_blur(gradient):

    blurred = cv2.GaussianBlur(gradient, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)

    return thresh

def image_morphology(thresh):
    # 建立一个椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # 执行图像形态学, 细节直接查文档，很简单
    #在这里我们选取ELLIPSE核，采用CLOSE操作
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    #从上图我们可以发现和原图对比，发现有细节丢失，这会干扰之后的昆虫轮廓的检测，
    # 要把它们扩充，分别执行4次形态学腐蚀与膨胀
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    return closed

#找到最大的矩形边框
def CutPicture(img):
    sizeY = np.mean(img,axis=0)#按列求均值
    indexY = np.argwhere(sizeY > 0)#找出大于0的列
    cutShapYmin = indexY[0]
    cutShapYmax = indexY[len(indexY)-1]

    sizeX = np.mean(img, axis=1)  # 按列求均值
    indexX = np.argwhere(sizeX > 0)  # 找出大于0的列
    cutShapXmin = indexX[0]
    cutShapXmax = indexX[len(indexX) - 1]

    cnt = np.array([[cutShapYmin,cutShapXmin], [cutShapYmax, cutShapXmin],
                    [cutShapYmin, cutShapXmax], [cutShapYmax, cutShapXmax]])  # 必须是array数组的形式
    #c = sorted(cnt, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(cnt)
    box = np.int0(cv2.boxPoints(rect))
    return box

#找到图像的轮廓
def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    (_, cnts, _) = cv2.findContours(closed.copy(),
       cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    return box

#画出有用图像的轮廓
def drawcnts_and_cut(original_img, box):
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
    # draw a bounding box arounded the detected barcode and display the image
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1 + hight, x1:x1 + width]
    return draw_img, crop_img

def walk():
    img_path = r'./png/0.png'
    save_path = r'./png/0s.png'
    original_img, gray = get_image(img_path)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    #box = findcnts_and_box_point(closed)
    box = CutPicture(closed)
    draw_img, crop_img = drawcnts_and_cut(original_img, box)

    cv2.imshow('original_img', original_img)
    cv2.imshow('blurred', blurred)
    cv2.imshow('gradX', gradX)
    cv2.imshow('gradY', gradY)
    cv2.imshow('final', gradient)
    cv2.imshow('thresh', thresh)
    cv2.imshow('closed', closed)

    cv2.imshow('draw_img', draw_img)
    cv2.imshow('crop_img', crop_img)
    cv2.waitKey(20171219)#等待用户输入，否则一直显示以上的图
    cv2.imwrite(save_path, crop_img)

def rewalk():
    img_path = r'./png/0s.png'
    save_path = r'./png/0ss.png'
    original_img, gray = get_image(img_path)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    #replicate = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    #边缘加宽，不怎么好用。
    cv2.imwrite(save_path, closed)

walk()
rewalk()