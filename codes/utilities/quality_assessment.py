import numpy as np
import cv2

# Mean Squared Error (MSE)
def mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Signal-to-Noise Ratio (SNR)
def snr(image1, image2):
    signal = np.sum(image1 ** 2)
    noise = np.sum((image1 - image2) ** 2)
    return 10 * np.log10(signal / noise)

# Peak Signal-to-Noise Ratio (PSNR)
def psnr(image1, image2):
    max_val = np.max(image1)
    return 10 * np.log10((max_val ** 2) / mse(image1, image2))

# Structural Similarity Index (SSIM)
def ssim_map(image1, image2, K1=0.01, K2=0.03, L=255):
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2

    mean_image1 = np.mean(image1)
    mean_image2 = np.mean(image2)
    cov = np.cov(image1, image2)

    numerator = (2 * mean_image1 * mean_image2 + c1) * (2 * cov + c2)
    denominator = ((mean_image1 ** 2 + mean_image2 ** 2 + c1) * (np.var(image1) + np.var(image2) + c2))

    ssim_map = numerator / denominator
    return ssim_map

def ssim(image1, image2, K1=0.01, K2=0.03, L=255):
    ssim_map_val = ssim_map(image1, image2, K1, K2, L)
    return np.mean(ssim_map_val)

# Root Mean Square Error (RMSE)
def rmse(image1, image2):
    return np.sqrt(np.mean((image1 - image2) ** 2))

# Mean Absolute Error (MAE)
def mae(image1, image2):
    return np.mean(np.abs(image1 - image2))

# Mean Structural Similarity Index (MSSIM)
def mssim(image1, image2):    
    # Define the scales for the MSSIM calculation
    scales = [0.5, 0.75, 1.0]
    mssim = []
    
    for scale in scales:
        # Resize the images to the current scale
        img1_resized = cv2.resize(image1, None, fx= scale, fy= scale, interpolation= cv2.INTER_AREA)
        img2_resized = cv2.resize(image2, None, fx= scale, fy= scale, interpolation= cv2.INTER_AREA)
        
        # Calculate the SSIM for the current scale
        ssim_value = ssim(img1_resized, img2_resized)
        mssim.append(ssim_value)
    
    # Return the average MSSIM value
    return np.mean(mssim)

# Visual Information Fidelity (VIF)
def vif(image1, image2):
    sigma_nsq = 2
    return np.sum((image1 * image2 + sigma_nsq) / (image1 ** 2 + image2 ** 2 + sigma_nsq))

# Feature Similarity Index (FSIM)
def fsim(image1, image2):
    return np.sum((2 * image1 * image2 + 0.01) / (image1 ** 2 + image2 ** 2 + 0.01))

# Multi-Scale Structural Similarity Index (MS-SSIM)
def ms_ssim(image1, image2):
    
    def downsample(image):
        return (image[::2, ::2] + image[1::2, ::2] + image[::2, 1::2] + image[1::2, 1::2]) / 4

    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    levels = len(weights)
    mssim_val = 1
    for i in range(levels):
        mssim_val *= mssim(image1, image2)
        if i < levels - 1:
            image1 = downsample(image1)
            image2 = downsample(image2)
    return mssim_val