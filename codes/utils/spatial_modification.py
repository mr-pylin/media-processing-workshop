import numpy as np

def rotate(img: np.ndarray, degree: float, expand: bool = False) -> np.ndarray:
    """
    Rotate an image by a specified degree.
    
    Args:
        img (np.ndarray): Input image to be rotated.
        degree (float): Rotation angle in degrees.
        expand (bool, optional): Whether to expand the image to fit the entire rotated content. Defaults to False.
    
    Returns:
        np.ndarray: Rotated image.
    """
    
    # define the rotation angle in radians
    rad = degree / 180 * np.pi
    total_row, total_col = img.shape
    transform_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    
    # determine center of the image
    old_center_idx = total_row // 2, total_col // 2
    
    if expand:
        # calculate new image dimensions
        new_r = int(total_row * abs(np.cos(rad)) + total_col * abs(np.sin(rad)))
        new_c = int(total_row * abs(np.sin(rad)) + total_col * abs(np.cos(rad)))
        new_center_idx = new_r // 2, new_c // 2
        rotated_image = np.zeros((new_r, new_c), dtype=img.dtype)
    else:
        new_r, new_c = total_row, total_col
        new_center_idx = old_center_idx
        rotated_image = np.zeros_like(img)
    
    # perform rotation
    for row in range(new_r):
        for col in range(new_c):
            idx = np.array([row - new_center_idx[0], col - new_center_idx[1]])
            new_row, new_col = (np.matmul(idx, transform_mat) + old_center_idx).astype(int)
            if 0 <= new_row < total_row and 0 <= new_col < total_col:
                rotated_image[row, col] = img[new_row, new_col]
    
    return rotated_image

def nearest_neighbor_interpolation(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resize an image using nearest neighbor interpolation.
    
    Args:
        image (np.ndarray): Input image.
        new_height (int): New height of the resized image.
        new_width (int): New width of the resized image.
    
    Returns:
        np.ndarray: Resized image using nearest neighbor interpolation.
    """
    
    # get the height and width of the input image
    height, width = image.shape[:2]
    
    # initialize the new image with zeros, preserving the number of color channels if present
    new_image = np.zeros((new_height, new_width, *image.shape[2:]))
    
    # calculate the ratio between old and new dimensions
    x_ratio = width / new_width
    y_ratio = height / new_height
    
    # loop over the new image dimensions
    for i in range(new_height):
        for j in range(new_width):
            # find the nearest neighbor in the original image
            x = int(j * x_ratio)
            y = int(i * y_ratio)
            
            # assign the pixel value from the original image to the new image
            new_image[i, j] = image[y, x]
    
    # return the resized image with the same data type as the original image
    return new_image.astype(image.dtype)

def bilinear_interpolation(image, new_height, new_width):
    """
    Resize an image using bilinear interpolation.
    
    Args:
        image (np.ndarray): Input image.
        new_height (int): New height of the resized image.
        new_width (int): New width of the resized image.
    
    Returns:
        np.ndarray: Resized image using bilinear interpolation.
    """
    
    # capable of handling rgb images
    height, width = image.shape[:2]
    new_image = np.zeros((new_height, new_width, *image.shape[2:]))
    
    x_ratio = float(width - 1) / new_width
    y_ratio = float(height - 1) / new_height
    
    for i in range(new_height):
        for j in range(new_width):
            x = j * x_ratio
            y = i * y_ratio
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
            
            dx = x - x1
            dy = y - y1
            
            new_image[i, j] = (1 - dx) * (1 - dy) * image[y1, x1] + dx * (1 - dy) * image[y1, x2] + (1 - dx) * dy * image[y2, x1] + dx * dy * image[y2, x2]
    
    return new_image.astype(image.dtype)

def fourier_transform_interpolation(image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """
    Resize an image using Fourier transform interpolation.
    
    Args:
        image (np.ndarray): Input image.
        new_height (int): New height of the resized image.
        new_width (int): New width of the resized image.
    
    Returns:
        np.ndarray: Resized image using Fourier transform interpolation.
    """
    
    old_shape = np.array(image.shape[:2])
    new_shape = np.array((new_height, new_width))
    
    # discrete fourier transform
    fft_image_old_shifted = np.fft.fftshift(np.fft.fft2(image))
    
    # prepare new shape array with zero padding
    fft_image_new_shifted = np.zeros(new_shape, dtype=np.complex128)
    
    # calculate the center of the old and new shapes
    old_center = old_shape // 2
    new_center = new_shape // 2
    
    # calculate the ranges for the new image
    start_y_new = new_center[0] - old_center[0]
    end_y_new = start_y_new + old_shape[0]
    start_x_new = new_center[1] - old_center[1]
    end_x_new = start_x_new + old_shape[1]
    
    # calculate the ranges for the old image
    start_y_old = max(0, old_center[0] - new_center[0])
    end_y_old = start_y_old + min(new_shape[0], old_shape[0])
    start_x_old = max(0, old_center[1] - new_center[1])
    end_x_old = start_x_old + min(new_shape[1], old_shape[1])
    
    # ensure indices are valid
    start_y_new = max(0, start_y_new)
    end_y_new = min(new_shape[0], end_y_new)
    start_x_new = max(0, start_x_new)
    end_x_new = min(new_shape[1], end_x_new)
    
    # copy the transformed data into the new array
    fft_image_new_shifted[start_y_new:end_y_new, start_x_new:end_x_new] = fft_image_old_shifted[
        start_y_old:end_y_old, start_x_old:end_x_old
    ]
    
    # inverse discrete fourier transform
    ifft_image_new = np.fft.ifft2(np.fft.ifftshift(fft_image_new_shifted)).real
    
    # normalize the output
    new_image = 255 * ((ifft_image_new - ifft_image_new.min()) / (ifft_image_new.max() - ifft_image_new.min()))
    
    return new_image.astype(np.uint8)

def histogram(img: np.ndarray, rng: int) -> tuple[np.ndarray]:
    """
    Calculate the histogram of a grayscale image.
    
    Args:
        img (np.ndarray): Input grayscale image.
        rng (int): Range of pixel values.
    
    Returns:
        tuple[np.ndarray]: Histogram of the input image.
    
    Raises:
        AssertionError: If the input image is not 2D.
    """
    
    assert img.ndim == 2, 'This function can only calculate histogram for 2d images.'
    
    idx, counts = np.unique(img, return_counts= True)
    hist = np.zeros(shape= rng, dtype= np.int64)
    hist[idx] = counts
    
    return hist

def histogram_scale(img: np.ndarray, lower_range, upper_range) -> np.ndarray:
    """
    Scale the histogram of an image to a specified range.
    
    Args:
        img (np.ndarray): Input image.
        lower_range (int): Lower bound of the new range.
        upper_range (int): Upper bound of the new range.
    
    Returns:
        np.ndarray: Image with scaled histogram.
    """
    
    # normalize -> [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # Rescale
    img = np.round(img * (upper_range - lower_range) + lower_range).astype(np.uint8)
    
    return img

def histogram_equalization(img: np.ndarray, ) -> np.ndarray:
    """
    Apply histogram equalization to an image.
    
    Args:
        img (np.ndarray): Input image.
    
    Returns:
        np.ndarray: Image with equalized histogram.
    """
    
    img_hist, _ = np.histogram(img, bins= 256, range= [0, 255])
    
    # calculate Cumulative Distribution Function (CDF)
    img_cdf = np.zeros_like(img_hist)
    
    for idx in range(len(img_hist)):
        img_cdf[idx] = img_cdf[idx - 1] + img_hist[idx]
    
    img_cdf = img_cdf / np.max(img_cdf)
    
    # rescale the result to range (0, 255)
    img_cdf = np.round(img_cdf * 255).astype(np.uint8)
    
    # map new values
    final_image = np.zeros_like(img)
    for intensity in range(256):
        final_image[img == intensity] = img_cdf[intensity]
    
    return histogram_scale(final_image, lower_range= 0, upper_range= 255)

def local_histogram_equalization(img: np.ndarray, windows_size: tuple[int, int]) -> np.ndarray:
    """
    Apply local histogram equalization to an image.
    
    Args:
        img (np.ndarray): Input image.
        window_size (tuple[int, int]): Size of the sliding window.
    
    Returns:
        np.ndarray: Image with locally equalized histogram.
    """
    
    total_row_cells = np.ceil(img.shape[0] / windows_size[0]).astype(np.int64)
    total_col_cells = np.ceil(img.shape[1] / windows_size[1]).astype(np.int64)
    new_img = np.zeros_like(img)
    
    for row_c in range(total_row_cells):
        for col_c in range(total_col_cells):
            h = row_c * windows_size[0]
            w = col_c * windows_size[1]
            new_img[h: h + windows_size[0], w: w + windows_size[1]] = histogram_equalization(img[h: h + windows_size[0], w: w + windows_size[1]])
    
    return new_img