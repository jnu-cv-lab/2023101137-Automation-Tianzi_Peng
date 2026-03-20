import cv2
import numpy as np 
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = cv2.imread('test_zmjjkk.jpg')

    height= img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    dtype = img.dtype
    print("图像基本信息：")
    print(f" 宽度 (width)：{width}")
    print(f" 高度 (height)：{height}")
    print(f" 通道数 (channels)：{channels}")
    print(f" 像素数据类型 (dtype)：{dtype}")

    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('Image.jpg', img)

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.imwrite('gray_example.jpg', gray_image)

    p=img[10, 15]
    print(f"像素 (15, 10) 的 BGR 值: {p}")

    crop = img[150:450, 350:650]
    cv2.imwrite('crop_example.jpg', crop)