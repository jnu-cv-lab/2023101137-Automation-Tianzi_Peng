import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_test_image():
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), 3)
    cv2.circle(img, (350, 100), 50, (0, 0, 255), 3)
    cv2.line(img, (50, 250), (250, 250), (0, 255, 0), 3)
    cv2.line(img, (50, 300), (250, 300), (0, 255, 0), 3)
    cv2.line(img, (350, 250), (450, 250), (0, 0, 0), 3)
    cv2.line(img, (400, 200), (400, 300), (0, 0, 0), 3)
    return img

test_img = create_test_image()
h, w = test_img.shape[:2]

# (A) 相似变换矩阵 (Similarity): 旋转 + 缩放 + 平移
theta = np.radians(30)
scale = 0.8
tx, ty = 100, 50
M_sim = np.array([
    [scale * np.cos(theta), -scale * np.sin(theta), tx],
    [scale * np.sin(theta),  scale * np.cos(theta), ty]
], dtype=np.float32)

# (B) 仿射变换矩阵 (Affine): 相似变换 + 错切 (Shear)
shear_x = 0.5
M_aff = np.array([
    [1, shear_x, 0],
    [0, 1,       0]
], dtype=np.float32)
M_aff[0, 2] = -70  # 添加平移以保持图像在视野内

# (C) 透视变换矩阵 (Perspective)
M_per = np.array([
    [1.2, 0.2, -50],
    [0.1, 1.1, -20],
    [0.0009, 0.0001, 1]
], dtype=np.float32)

img_sim = cv2.warpAffine(test_img, M_sim, (w, h), borderValue=(255, 255, 255))
img_aff = cv2.warpAffine(test_img, M_aff, (w, h), borderValue=(255, 255, 255))
img_per = cv2.warpPerspective(test_img, M_per, (w, h), borderValue=(255, 255, 255))

titles = ['Original', 'Similarity', 'Affine', 'Perspective']
images = [test_img, img_sim, img_aff, img_per]
plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('lab05_work1.jpg')