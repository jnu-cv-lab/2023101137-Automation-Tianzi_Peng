import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_getPerspectiveTransform(src, dst):
    A = np.zeros((8, 8), dtype=np.float64)
    B = np.zeros((8, 1), dtype=np.float64)
    
    for i in range(4):
        x, y = src[i][0], src[i][1]
        u, v = dst[i][0], dst[i][1]
        
        A[2*i] = [x, y, 1, 0, 0, 0, -x*u, -y*u]
        B[2*i] = u

        A[2*i+1] = [0, 0, 0, x, y, 1, -x*v, -y*v]
        B[2*i+1] = v
        
    h = np.linalg.solve(A, B)
    h = np.append(h, 1.0)
    H = h.reshape((3, 3))
    
    return H

img = cv2.imread('test.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

pts_src = np.array([
    [584, 257],   # 左上角
    [1240, 382],   # 右上角
    [912, 1373],  # 右下角
    [0, 944]    # 左下角
], dtype=np.float32)

width = 210 * 3
height = 297 * 3
pts_dst = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
warped_img1 = cv2.warpPerspective(img, M, (width, height))
print("透视变换矩阵 M:\n", M)

H= my_getPerspectiveTransform(pts_src, pts_dst)
wraped_img2 = cv2.warpPerspective(img, H, (width, height))
print("手动计算的透视变换矩阵 H:\n", H)

plt.imshow(cv2.cvtColor(warped_img1, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('work2_save.jpg', warped_img1)
plt.imshow(cv2.cvtColor(wraped_img2, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite('work2_save_manual.jpg', wraped_img2)