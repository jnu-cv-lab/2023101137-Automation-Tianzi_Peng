# WEEK01  
- 读取图片，转换到YCrCb色彩空间，再分离成Y、Cr、Cb三个单独通道  
- Y、Cr、Cb三个单独通道单独显示的话应该是黑白的，因为只是二维数组  
- 对Cr、Cb通道下采样（就是变回马赛克），我做的是宽度和高度整除4  
```  
Cr_sub = cv2.resize(Cr, (w//4, h//4), interpolation=cv2.INTER_AREA)  
Cb_sub = cv2.resize(Cb, (w//4, h//4), interpolation=cv2.INTER_AREA)  
```  
- 通过插值重建YCrCb，比较计算PSNR（Peak Signal-to-Noise Ratio，峰值信噪比），这是最常用的图像重建/压缩质量客观评价指标  
