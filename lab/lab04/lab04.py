import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_chirp(size=512):
    """生成 Chirp 测试图 """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    chirp = np.cos(100 * np.pi * R**2)
    chirp = ((chirp + 1) * 127.5).astype(np.uint8)
    return chirp

def generate_checkerboard(size=512, block_size=16):
    """生成棋盘格测试图"""
    indices = np.indices((size, size))
    checker = ((indices[0] // block_size) + (indices[1] // block_size)) % 2
    return (checker * 255).astype(np.uint8)

def get_fft_spectrum(img):
    """计算并返回图像的频域中心化对数频谱图"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5) # 加上一个极小值防止log(0)
    return magnitude_spectrum

print("运行第一部分...")
img_chirp = generate_chirp(512)
M = 4 

downsampled_direct = img_chirp[::M, ::M]

sigma_optimal = 0.45 * M  
img_blurred = cv2.GaussianBlur(img_chirp, (0, 0), sigmaX=sigma_optimal)
downsampled_anti = img_blurred[::M, ::M]

fft_original = get_fft_spectrum(img_chirp)
fft_blurred = get_fft_spectrum(img_blurred)

plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(img_chirp, cmap='gray'), plt.title('Original Chirp')
plt.subplot(232), plt.imshow(downsampled_direct, cmap='gray'), plt.title('Direct Downsample (Aliasing)')
plt.subplot(233), plt.imshow(downsampled_anti, cmap='gray'), plt.title('Filtered Downsample (Anti-aliasing)')
plt.subplot(234), plt.imshow(fft_original, cmap='gray'), plt.title('FFT Original')
plt.subplot(235), plt.imshow(fft_blurred, cmap='gray'), plt.title('FFT after Gaussian (High-freq cut)')
plt.tight_layout()
plt.savefig('lab04_part1.png')
plt.show()

print("运行第二部分...")
sigmas = [0.5, 1.0, 1.8, 4.0] 
plt.figure(figsize=(16, 4))

for i, sigma in enumerate(sigmas):
    blurred = cv2.GaussianBlur(img_chirp, (0, 0), sigmaX=sigma)
    down = blurred[::M, ::M]
    
    plt.subplot(1, 4, i+1)
    plt.imshow(down, cmap='gray')
    
    if sigma < 1.8:
        desc = "(Under-smoothed, Aliasing)"
    elif sigma == 1.8:
        desc = "(Optimal = 0.45M)"
    else:
        desc = "(Over-smoothed)"
    plt.title(f'$\sigma$={sigma}\n{desc}')

plt.tight_layout()
plt.savefig('lab04_part2.png')
plt.show()
