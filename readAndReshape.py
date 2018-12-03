from PIL import Image#python3中image要从PIL中导入
import numpy as np

def getTestImgArray(filename,imgHeight,imgWidth):
    im = Image.open(filename)
    x_s = imgWidth
    y_s = imgHeight
    im_arr = im.resize((x_s, y_s), Image.ANTIALIAS)

    nm = im_arr.reshape((1, imgHeight*imgHeight))

    nm = nm.astype(np.float32)

    return nm