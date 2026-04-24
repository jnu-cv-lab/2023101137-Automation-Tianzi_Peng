import cv2
import numpy as np

def run_orb_pipeline(img1_path, img2_path, nfeatures=1000, save_prefix="ORB"):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)

    # ================= 任务 1：特征检测与描述 =================
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    img1_kp = cv2.drawKeypoints(img1_color, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2_color, kp2, None, color=(0, 255, 0))
    cv2.imwrite(f"{save_prefix}_{nfeatures}_kp1.png", img1_kp)
    cv2.imwrite(f"{save_prefix}_{nfeatures}_kp2.png", img2_kp)

    print(f"--- {save_prefix} (nfeatures={nfeatures}) ---")
    print(f"模板图关键点数: {len(kp1)}, 场景图关键点数: {len(kp2)}")
    if des1 is not None:
        print(f"描述子维度: {des1.shape[1]} bytes")

    # ================= 任务 2：特征匹配 =================
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(matches)
    
    img_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f"{save_prefix}_{nfeatures}_matches_top50.png", img_matches)
    print(f"总匹配数量: {total_matches}")

    # ================= 任务 3：RANSAC 剔除错误匹配 & 任务 4：目标定位 =================
    inlier_ratio = 0
    inliers_count = 0
    is_success = False

    if len(matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            matchesMask = mask.ravel().tolist()
            inliers_count = np.sum(mask)
            inlier_ratio = inliers_count / total_matches
            
            print(f"RANSAC 内点数量: {inliers_count}")
            print(f"内点比例: {inlier_ratio:.4f}")
            
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            img_ransac = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches, None, **draw_params)
            cv2.imwrite(f"{save_prefix}_{nfeatures}_matches_ransac.png", img_ransac)

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            img2_box = cv2.polylines(img2_color.copy(), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imwrite(f"{save_prefix}_{nfeatures}_localization.png", img2_box)
            is_success = True
            print("定位状态: 成功")
        else:
            print("定位状态: 失败 (未能计算出单应性矩阵)")
    else:
        print("匹配点不足以计算 Homography。")
        
    return len(kp1), len(kp2), total_matches, inliers_count, inlier_ratio, is_success

def run_sift_pipeline(img1_path, img2_path):
    # ================= 选做任务：SIFT 特征匹配 =================
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img1_color = cv2.imread(img1_path)
    img2_color = cv2.imread(img2_path)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    total_matches = len(good_matches)
    print(f"\n--- SIFT 实验 ---")
    print(f"Lowe 测试后匹配数量: {total_matches}")

    if total_matches > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            inliers_count = np.sum(mask)
            inlier_ratio = inliers_count / total_matches
            print(f"RANSAC 内点数量: {inliers_count}")
            print(f"内点比例: {inlier_ratio:.4f}")
            
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            img2_box = cv2.polylines(img2_color.copy(), [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imwrite("SIFT_localization.png", img2_box)
            print("SIFT 定位状态: 成功")
        else:
            print("SIFT 定位状态: 失败")

if __name__ == "__main__":
    img1_file = 'box.png'
    img2_file = 'box_in_scene.png'
    
    for n in [500, 1000, 2000]:
        run_orb_pipeline(img1_file, img2_file, nfeatures=n)
        
    run_sift_pipeline(img1_file, img2_file)