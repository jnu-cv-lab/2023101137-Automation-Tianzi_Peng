import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('test_zmjjkk.jpg',cv.IMREAD_GRAYSCALE)
h = img.shape[0]
w = img.shape[1]
blur = cv.GaussianBlur(img, (5, 5), 0)   
sub1 = cv.resize(img, (w//4, h//4), interpolation=cv.INTER_AREA)
sub2 = cv.resize(blur, (w//4, h//4), interpolation=cv.INTER_AREA)
NEAREST_up = cv.resize(sub1, (w, h), interpolation=cv.INTER_NEAREST)
LINEAR_up = cv.resize(sub1, (w, h), interpolation=cv.INTER_LINEAR)
CUBIC_up = cv.resize(sub1, (w, h), interpolation=cv.INTER_CUBIC)

mse1 = ((img - NEAREST_up) ** 2).mean()
psnr1 = cv.PSNR(img, NEAREST_up)
mse2 = ((img - LINEAR_up) ** 2).mean()
psnr2 = cv.PSNR(img, LINEAR_up)
mse3 = ((img - CUBIC_up) ** 2).mean()
psnr3 = cv.PSNR(img, CUBIC_up)

print(f" NEAREST: MSE = {mse1:.4f}, PSNR = {psnr1:.4f} dB")
print(f" LINEAR: MSE = {mse2:.4f}, PSNR = {psnr2:.4f} dB")
print(f" CUBIC: MSE = {mse3:.4f}, PSNR = {psnr3:.4f} dB")

dct_img = cv.dct(np.float32(img))
dct_linear = cv.dct(np.float32(LINEAR_up))
dct_nearst = cv.dct(np.float32(NEAREST_up))
dct_cubic = cv.dct(np.float32(CUBIC_up))

dct_img_log = 20 * np.log(np.abs(dct_img) + 1)
dct_linear_log = 20 * np.log(np.abs(dct_linear) + 1)
dct_nearst_log = 20 * np.log(np.abs(dct_nearst) + 1)
dct_cubic_log = 20 * np.log(np.abs(dct_cubic) + 1)

total_energy_img = np.sum(dct_img**2)
low_freq_energy_img = np.sum(dct_img[:h//4, :w//4]**2)
ratio_img = low_freq_energy_img / total_energy_img

total_energy_linear = np.sum(dct_linear**2)
low_freq_energy_linear = np.sum(dct_linear[:h//4, :w//4]**2)
ratio_linear = low_freq_energy_linear / total_energy_linear

total_energy_nearst = np.sum(dct_nearst**2)
low_freq_energy_nearst = np.sum(dct_nearst[:h//4, :w//4]**2)
ratio_nearst = low_freq_energy_nearst / total_energy_nearst

total_energy_cubic = np.sum(dct_cubic**2)
low_freq_energy_cubic = np.sum(dct_cubic[:h//4, :w//4]**2)
ratio_cubic = low_freq_energy_cubic / total_energy_cubic

print(f"Img low-frequency energy ratio: {ratio_img:.4f}")
print(f"LINEAR_up low-frequency energy ratio: {ratio_linear:.4f}")
print(f"NEAREST_up low-frequency energy ratio: {ratio_nearst:.4f}")
print(f"CUBIC_up low-frequency energy ratio: {ratio_cubic:.4f}")


# dft1 = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift1 = np.fft.fftshift(dft1)
# magnitude_spectrum1 = 20 * np.log(cv.magnitude(dft_shift1[:,:,0], dft_shift1[:,:,1]))
# dft2 = cv.dft(np.float32(sub1), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift2 = np.fft.fftshift(dft2)
# magnitude_spectrum2 = 20 * np.log(cv.magnitude(dft_shift2[:,:,0], dft_shift2[:,:,1]))
# dft3 = cv.dft(np.float32(LINEAR_up), flags=cv.DFT_COMPLEX_OUTPUT)
# dft_shift3 = np.fft.fftshift(dft3)
# magnitude_spectrum3 = 20 * np.log(cv.magnitude(dft_shift3[:,:,0], dft_shift3[:,:,1]))

f1 = np.fft.fft2(img)
fshift1 = np.fft.fftshift(f1)
magnitude_spectrum1 = 20*np.log(np.abs(fshift1))
f2 = np.fft.fft2(sub1)
fshift2 = np.fft.fftshift(f2)
magnitude_spectrum2 = 20*np.log(np.abs(fshift2))
f3 = np.fft.fft2(LINEAR_up)
fshift3 = np.fft.fftshift(f3)
magnitude_spectrum3 = 20*np.log(np.abs(fshift3))

cv.imshow('img',img)
cv.imshow('sub1',sub1)
cv.imshow('sub2',sub2)
cv.imshow('NEAREST_up',NEAREST_up)
cv.imshow('LINEAR_up',LINEAR_up)
cv.imshow('CUBIC_up',CUBIC_up)
cv.imshow('magnitude_spectrum1', magnitude_spectrum1.astype(np.uint8))
cv.imshow('magnitude_spectrum2', magnitude_spectrum2.astype(np.uint8))
cv.imshow('magnitude_spectrum3', magnitude_spectrum3.astype(np.uint8))
cv.imshow('DCT_img_log', dct_img_log.astype(np.uint8))
cv.imshow('DCT_linear_log', dct_linear_log.astype(np.uint8))
cv.imwrite('img.jpg', img)
cv.imwrite('sub1.jpg', sub1)
cv.imwrite('sub2.jpg', sub2)
cv.imwrite('NEAREST_up.jpg', NEAREST_up)
cv.imwrite('LINEAR_up.jpg', LINEAR_up)
cv.imwrite('CUBIC_up.jpg', CUBIC_up)
cv.imwrite('magnitude_spectrum1.jpg', magnitude_spectrum1.astype(np.uint8))
cv.imwrite('magnitude_spectrum2.jpg', magnitude_spectrum2.astype(np.uint8))
cv.imwrite('magnitude_spectrum3.jpg', magnitude_spectrum3.astype(np.uint8))
cv.imwrite('DCT_img_log.jpg', dct_img_log.astype(np.uint8))
cv.imwrite('DCT_linear_log.jpg', dct_linear_log.astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows
