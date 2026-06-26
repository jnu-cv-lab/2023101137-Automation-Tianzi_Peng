import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9, 6)

SQUARE_SIZE = 25.0

IMAGE_DIR = "images"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
              glob.glob(os.path.join(IMAGE_DIR, "*.png")) + \
              glob.glob(os.path.join(IMAGE_DIR, "*.jpeg"))

if len(image_paths) == 0:
    raise ValueError("没有找到图片，请把标定图片放到 images 文件夹中。")

print(f"共读取到 {len(image_paths)} 张图片")

gray_shape = None
success_count = 0

for idx, path in enumerate(image_paths):
    img = cv2.imread(path)

    if img is None:
        print(f"无法读取图片：{path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_shape = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        success_count += 1
        objpoints.append(objp)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners_subpix)

        img_draw = img.copy()
        cv2.drawChessboardCorners(img_draw, CHECKERBOARD, corners_subpix, ret)

        save_path = os.path.join(RESULT_DIR, f"corners_{idx + 1}.jpg")
        cv2.imwrite(save_path, img_draw)

        print(f"[成功] {path}，角点结果保存为 {save_path}")

    else:
        print(f"[失败] {path}，未检测到完整棋盘格角点")

print(f"\n成功检测角点的图片数量：{success_count}")

if success_count < 5:
    raise ValueError("有效图片太少，建议至少保证 15 张图片中有较多图片能成功检测角点。")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray_shape,
    None,
    None
)

print("\n========== 标定结果 ==========")
print("相机内参矩阵 K：")
print(camera_matrix)

print("\n畸变参数 D = [k1, k2, p1, p2, k3]：")
print(dist_coeffs.ravel())

print("\n每张图片的外参：")
for i in range(len(rvecs)):
    print(f"\n第 {i + 1} 张图片：")
    print("旋转向量 rvec:")
    print(rvecs[i].ravel())
    print("平移向量 tvec:")
    print(tvecs[i].ravel())

total_error = 0

for i in range(len(objpoints)):
    imgpoints_projected, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
    total_error += error

mean_error = total_error / len(objpoints)

print("\n========== 重投影误差 ==========")
print(f"平均重投影误差：{mean_error:.4f} pixel")

test_img = cv2.imread(image_paths[0])
h, w = test_img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

undistorted = cv2.undistort(
    test_img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

x, y, w_roi, h_roi = roi
if w_roi > 0 and h_roi > 0:
    undistorted_crop = undistorted[y:y + h_roi, x:x + w_roi]
else:
    undistorted_crop = undistorted

cv2.imwrite(os.path.join(RESULT_DIR, "original.jpg"), test_img)
cv2.imwrite(os.path.join(RESULT_DIR, "undistorted.jpg"), undistorted)
cv2.imwrite(os.path.join(RESULT_DIR, "undistorted_crop.jpg"), undistorted_crop)

print("\n========== 去畸变结果 ==========")
print("原图已保存：results/original.jpg")
print("去畸变图已保存：results/undistorted.jpg")
print("裁剪后的去畸变图已保存：results/undistorted_crop.jpg")

with open(os.path.join(RESULT_DIR, "calibration_result.txt"), "w", encoding="utf-8") as f:
    f.write("相机内参矩阵 K:\n")
    f.write(str(camera_matrix))
    f.write("\n\n畸变参数 D = [k1, k2, p1, p2, k3]:\n")
    f.write(str(dist_coeffs.ravel()))
    f.write("\n\n平均重投影误差:\n")
    f.write(f"{mean_error:.4f} pixel\n")

print("\n标定结果已保存：results/calibration_result.txt")