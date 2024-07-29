import numpy as np

def rotate(img: np.ndarray, degree: float, expand: bool = False) -> np.ndarray:
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
    # capable of handling rgb images
    height, width = image.shape[:2]
    new_image = np.zeros((new_height, new_width, *image.shape[2:]))

    x_ratio = width / new_width
    y_ratio = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x = int(j * x_ratio)
            y = int(i * y_ratio)
            new_image[i, j] = image[y, x]

    return new_image

def bilinear_interpolation(image, new_height, new_width):
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

    return new_image

def histogram(img: np.ndarray, rng: int) -> tuple[np.ndarray]:
    assert img.ndim == 2, 'This function can only calculate histogram for 2d images.'

    idx, counts = np.unique(img, return_counts= True)
    hist = np.zeros(shape= rng, dtype= np.int64)
    hist[idx] = counts

    return hist

def histogram_scale(img: 'np.ndarray', lower_range, upper_range) -> 'np.ndarray':
    # normalize -> [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # Rescale
    img = np.round(img * (upper_range - lower_range) + lower_range).astype(np.uint8)

    return img

def histogram_equalization(img: np.ndarray, ) -> np.ndarray:
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
    total_row_cells = np.ceil(img.shape[0] / windows_size[0]).astype(np.int64)
    total_col_cells = np.ceil(img.shape[1] / windows_size[1]).astype(np.int64)
    new_img = np.zeros_like(img)

    for row_c in range(total_row_cells):
        for col_c in range(total_col_cells):
            h = row_c * windows_size[0]
            w = col_c * windows_size[1]
            new_img[h: h + windows_size[0], w: w + windows_size[1]] = histogram_equalization(img[h: h + windows_size[0], w: w + windows_size[1]])
    
    return new_img