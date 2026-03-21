import cv2
import numpy as np

if __name__ == "__main__":
    img = cv2.imread('test_zmjjkk.jpg')

    YCrCb=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    Y=YCrCb[:,:,0]
    Cr=YCrCb[:,:,1]
    Cb=YCrCb[:,:,2]
    
    cv2.imshow('YCrCb',YCrCb)
    cv2.imshow('Y', Y)
    cv2.imshow('Cr', Cr)
    cv2.imshow('Cb', Cb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    h, w = Y.shape

    Cr_sub = cv2.resize(Cr, (w//4, h//4), interpolation=cv2.INTER_AREA)
    Cb_sub = cv2.resize(Cb, (w//4, h//4), interpolation=cv2.INTER_AREA)

    Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_NEAREST)
    Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_NEAREST)

    YCrCb_reconstructed = np.zeros_like(YCrCb)
    YCrCb_reconstructed[:,:,0] = Y
    YCrCb_reconstructed[:,:,1] = Cr_up
    YCrCb_reconstructed[:,:,2] = Cb_up

    img_reconstructed = cv2.cvtColor(YCrCb_reconstructed, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('img',img)
    cv2.imshow('YCrCb',YCrCb)
    cv2.imshow('YCrCb_reconstructed',YCrCb_reconstructed)
    cv2.imshow('img_reconstructed',img_reconstructed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    psnr = cv2.PSNR(img, img_reconstructed)
    print(f"PSNR (cv2) = {psnr:.4f} dB")