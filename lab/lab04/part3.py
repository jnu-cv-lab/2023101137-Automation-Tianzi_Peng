import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_test_photo(size=512):
    """加载测试合成图"""
    img = np.zeros((size, size), dtype=np.uint8)
    # 左半部分平滑渐变
    y, x = np.indices((size, size//2))
    img[:, :size//2] = (x + y) * (255.0 / (size * 1.5))
    # 右半部分密集条纹
    for i in range(0, size, 4):
        img[:, size//2 + i:size//2 + i+2] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.circle(img, (size//2, size//2), 80, (255,), -1)
    cv2.circle(img, (size//2, size//2), 60, (0,), -1)
    return img

M = 4
sigma_optimal = 0.45 * M  

img_photo = get_test_photo(512)

img_uniform_blurred = cv2.GaussianBlur(img_photo, (0, 0), sigmaX=sigma_optimal)
down_uniform = img_uniform_blurred[::M, ::M]

img_float = img_photo.astype(np.float32)

grad_x = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = cv2.magnitude(grad_x, grad_y)

grad_mag_smoothed = cv2.GaussianBlur(grad_mag, (15, 15), sigmaX=5)
alpha_grad = grad_mag_smoothed / grad_mag_smoothed.max()

img_adaptive_grad = (img_float * (1 - alpha_grad) + img_uniform_blurred.astype(np.float32) * alpha_grad).astype(np.uint8)
down_adaptive = img_adaptive_grad[::M, ::M]

plt.figure(figsize=(16, 9))

plt.subplot(231), plt.imshow(img_photo, cmap='gray'), plt.title('Original Photo')
plt.subplot(232), plt.imshow(alpha_grad, cmap='jet'), plt.title('Gradient Mask\n(Notice the "holes" in stripes)')
plt.subplot(233), plt.imshow(down_uniform, cmap='gray'), plt.title('Uniform Downsampling')

plt.subplot(234), plt.imshow(down_adaptive, cmap='gray'), plt.title('Adaptive (Gradient) Downsampling')

roi_y1, roi_x1 = 200, 230
size_roi = 80
roi_u1 = down_uniform[roi_y1//M:roi_y1//M+size_roi//M, roi_x1//M:roi_x1//M+size_roi//M]
roi_a1 = down_adaptive[roi_y1//M:roi_y1//M+size_roi//M, roi_x1//M:roi_x1//M+size_roi//M]

plt.subplot(235),plt.imshow(np.hstack((roi_u1, roi_a1)), cmap='gray'),plt.title('Center Detail\nLeft: Uniform | Right: Adaptive(Grad)')
plt.tight_layout()
plt.savefig('lab04_part3.png')
plt.show()