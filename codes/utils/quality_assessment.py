import numpy as np
import cv2


def mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Mean squared error between the two images.
    """
    
    return np.mean((image1 - image2) ** 2)


def snr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) between two images.
    
    Args:
        image1 (np.ndarray): Original image (signal).
        image2 (np.ndarray): Noisy image.
    
    Returns:
        float: Signal-to-noise ratio in decibels.
    """
    
    signal = np.sum(image1 ** 2)
    noise = np.sum((image1 - image2) ** 2)
    return 10 * np.log10(signal / noise)


def psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        image1 (np.ndarray): Reference image.
        image2 (np.ndarray): Distorted image.
    
    Returns:
        float: Peak signal-to-noise ratio in decibels.
    """
    
    max_val = np.max(image1)
    return 10 * np.log10((max_val ** 2) / mse(image1, image2))


def ssim_map(image1: np.ndarray, image2: np.ndarray, K1: float = 0.01, K2: float = 0.03, L: int = 255) -> np.ndarray:
    """
    Calculate the Structural Similarity Index Map (SSIM) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
        K1 (float, optional): Default is 0.01.
        K2 (float, optional): Default is 0.03.
        L (int, optional): Dynamic range of the pixel values. Defaults to 255.
    
    Returns:
        np.ndarray: SSIM map between the two images.
    """
    
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    
    mean_image1 = np.mean(image1)
    mean_image2 = np.mean(image2)
    cov = np.cov(image1, image2)
    
    numerator = (2 * mean_image1 * mean_image2 + c1) * (2 * cov + c2)
    denominator = ((mean_image1 ** 2 + mean_image2 ** 2 + c1) * (np.var(image1) + np.var(image2) + c2))
    
    ssim_map = numerator / denominator
    return ssim_map


def ssim(image1: np.ndarray, image2: np.ndarray, K1: float = 0.01, K2: float = 0.03, L: int = 255) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
        K1 (float, optional): Default is 0.01.
        K2 (float, optional): Default is 0.03.
        L (int, optional): Dynamic range of the pixel values. Defaults to 255.
    
    Returns:
        float: Average structural similarity index between the two images.
    """
    
    ssim_map_val = ssim_map(image1, image2, K1, K2, L)
    return np.mean(ssim_map_val)


def rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Root mean square error between the two images.
    """
    
    return np.sqrt(np.mean((image1 - image2) ** 2))


def mae(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Mean absolute error between the two images.
    """
    
    return np.mean(np.abs(image1 - image2))


def mssim(image1: np.ndarray, image2: np.ndarray) -> float:  
    """
    Calculate the Multi-Scale Structural Similarity Index (MSSIM) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Multi-scale structural similarity index between the two images.
    """
    
    # define the scales for the MSSIM calculation
    scales = [0.5, 0.75, 1.0]
    mssim = []
    
    for scale in scales:
        # resize the images to the current scale
        img1_resized = cv2.resize(image1, None, fx= scale, fy= scale, interpolation= cv2.INTER_AREA)
        img2_resized = cv2.resize(image2, None, fx= scale, fy= scale, interpolation= cv2.INTER_AREA)
        
        # calculate the SSIM for the current scale
        ssim_value = ssim(img1_resized, img2_resized)
        mssim.append(ssim_value)
    
    # return the average MSSIM value
    return np.mean(mssim)


def vif(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Visual Information Fidelity (VIF) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Visual information fidelity between the two images.
    """
    
    sigma_nsq = 2
    return np.sum((image1 * image2 + sigma_nsq) / (image1 ** 2 + image2 ** 2 + sigma_nsq))


def fsim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Feature Similarity Index (FSIM) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Feature similarity index between the two images.
    """
    
    return np.sum((2 * image1 * image2 + 0.01) / (image1 ** 2 + image2 ** 2 + 0.01))


def ms_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two images.
    
    Args:
        image1 (np.ndarray): First image.
        image2 (np.ndarray): Second image.
    
    Returns:
        float: Multi-scale structural similarity index between the two images.
    """
    
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