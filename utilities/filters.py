import numpy as np

def ideal(size: tuple[int], threshold: int, mode: str = 'low', shape: str = 'circle', pos: str = 'center', norm: bool = False) -> np.ndarray:
    """
    Create a 2D ideal low/high pass filter.

    Parameters
    ----------
    size : tuple[int]
        Size of the filter.
        e.g. (256, 256)
    
    threshold : int
        Cutoff Frequency or the amount of area with value=1.
        e.g. 32

    mode : {'low', 'high'}, default: 'low'
        Whether to create a low-pass or high-pass filter

    shape : {'circle', 'square'}, default: 'circle'
        Shape of the filter.

    pos : {'center', 'corner'}, default: 'center'
        The position of the filter.

    norm : bool, default: False
        Normalize the values to be accumulated to 1 [useful in convolution]
        Note: Do not normalize it for frequency multiplication e.g. np.fft.fft2
    
    Returns
    -------
    numpy.ndarray
    """

    # create a centered circled ideal low-pass filter
    if shape == 'circle':
        row_center, column_center = (size[0] - 1) / 2, (size[1] - 1) / 2
        x, y = np.meshgrid(np.arange(size[1]) - column_center, np.arange(size[0]) - row_center)
        distance = np.sqrt(x ** 2 + y ** 2)
        ideal_filter = np.where(distance <= threshold, 1., 0.)

    # create a centered squared low-pass filter
    elif shape == 'square':

        # if the filter does not have center, add an offest to make it fully centered
        offset = [0, 0]
        for i in range(2):
            if size[i] % 2 != 0:
                offset[i] = 1
        row_center, column_center = size[0] // 2, size[1] // 2
        ideal_filter = np.zeros(shape= size, dtype= np.float64)
        ideal_filter[row_center - threshold: row_center + threshold + offset[0], column_center - threshold: column_center + threshold + offset[1]] = 1
        
    else:
        raise ValueError(f"Invalid `shape` value: {shape}; should be 'circle' or 'square'.")

    if pos == 'corner':
        ideal_filter = np.fft.fftshift(ideal_filter)
    elif pos != 'center':
        raise ValueError(f"Invalid `pos` value: {pos}; should be 'center' or 'corner'.")

    if mode == 'high':
        ideal_filter = 1 - ideal_filter
    elif mode != 'low':
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'low' or 'high'.")

    if norm:
        ideal_filter /= np.sum(ideal_filter)

    return ideal_filter


def gaussian(size: tuple[int], sigma: float, mode: str = 'low', pos: str = 'center', norm: bool = False) -> np.ndarray:
    """
    Create a 2D Gaussian low/high pass filter.

    Parameters
    ----------
    size : tuple[int]
        Size of the filter.
        e.g. (256, 256)
    
    sigma : float
        Cutoff Frequency or standard deviation.
        e.g. 32

    mode : {'low', 'high'}, default: 'low'
        Whether to create a low-pass or high-pass filter

    pos : {'center', 'corner'}, default: 'center'
        The position of the filter.

    norm : bool, default: False
        Normalize the values to be accumulated to 1 [useful in convolution]
        Note: Do not normalize it for frequency multiplication e.g. np.fft.fft2
    
    Returns
    -------
    numpy.ndarray
    """

    row_center, column_center = size[0] // 2, size[1] // 2
    gaussian_filter = np.zeros(shape= size)
    for x in range(size[0]):
        for y in range(size[1]):
            gaussian_filter[x, y] = np.exp(-((x - row_center) ** 2 + (y - column_center) ** 2) / (2 * sigma ** 2))

    if pos == 'corner':
        gaussian_filter = np.fft.fftshift(gaussian_filter)
    elif pos != 'center':
        raise ValueError(f"Invalid `pos` value: {pos}; should be 'center' or 'corner'.")

    if mode == 'high':
        gaussian_filter = 1 - gaussian_filter
    elif mode != 'low':
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'low' or 'high'.")

    if norm:
        gaussian_filter /= np.sum(gaussian_filter)

    return gaussian_filter


def sinc(size: tuple[int], sigma: float, mode: str = 'low', pos: str = 'center', norm: bool = False) -> np.ndarray:
    """
    Create a 2D Sinus Cardinalis (sinc) low/high pass filter.

    Parameters
    ----------
    size : tuple[int]
        Size of the filter.
        e.g. (256, 256)
    
    sigma : float
        Cutoff Frequency or standard deviation.
        e.g. 32

    mode : {'low', 'high'}, default: 'low'
        Whether to create a low-pass or high-pass filter

    pos : {'center', 'corner'}, default: 'center'
        The position of the filter.

    norm : bool, default: False
        Normalize the values to be accumulated to 1 [useful in convolution]
        Note: Do not normalize it for frequency multiplication e.g. np.fft.fft2
    
    Returns
    -------
    numpy.ndarray
    """

    # create a centered sinc low-pass filter
    row_center, column_center = size[0] // 2, size[1] // 2
    x, y = np.meshgrid(np.arange(size[0]) - row_center, np.arange(size[1]) - column_center)
    distance = np.sqrt(x ** 2 + y ** 2)
    sinc_filter = np.sinc(distance / np.pi / sigma)

    # normalize in the range [0, 1]
    sinc_filter = (sinc_filter - sinc_filter.min()) / (sinc_filter.max() - sinc_filter.min())

    if pos == 'corner':
        sinc_filter = np.fft.fftshift(sinc_filter)
    elif pos != 'center':
        raise ValueError(f"Invalid `pos` value: {pos}; should be 'center' or 'corner'.")

    if mode == 'high':
        sinc_filter = 1 - sinc_filter
    elif mode != 'low':
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'low' or 'high'.")

    if norm:
        sinc_filter /= np.sum(sinc_filter)

    return sinc_filter


def butterworth(size: tuple[int], cutoff: float, n: int, mode: str = 'low', pos: str = 'center', norm: bool = False) -> np.ndarray:
    """
    Create a 2D ButterWorth low/high pass filter.

    Parameters
    ----------
    size : tuple[int]
        Size of the filter.
        e.g. (256, 256)
    
    cutoff : float
        Cutoff Frequency
        e.g. 32

    n : int
        The order [degree] of ButterWorth fitler.
        e.g. 2

    mode : {'low', 'high'}, default: 'low'
        Whether to create a low-pass or high-pass filter

    pos : {'center', 'corner'}, default: 'center'
        The position of the filter.

    norm : bool, default: False
        Normalize the values to be accumulated to 1 [useful in convolution]
        Note: Do not normalize it for frequency multiplication e.g. np.fft.fft2
    
    Returns
    -------
    numpy.ndarray
    """

    rows, cols = size
    x = np.linspace(-0.5, 0.5, cols) * cols
    y = np.linspace(-0.5, 0.5, rows) * rows
    radius = np.sqrt((x ** 2)[np.newaxis] + (y ** 2)[:, np.newaxis])
    butterworth_filter = 1 / (1 + (radius / cutoff) ** (2 * n))

    if pos == 'corner':
        butterworth_filter = np.fft.fftshift(butterworth_filter)
    elif pos != 'center':
        raise ValueError(f"Invalid `pos` value: {pos}; should be 'center' or 'corner'.")

    if mode == 'high':
        butterworth_filter = 1 - butterworth_filter
    elif mode != 'low':
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'low' or 'high'.")

    if norm:
        butterworth_filter /= np.sum(butterworth_filter)

    return butterworth_filter


def chebyshev(size: tuple[int], cutoff: float, n: int, mode: str = 'low', pos: str = 'center', norm: bool = False) -> np.ndarray:
    """
    Create a 2D Chebyshev low/high pass filter.

    Parameters
    ----------
    size : tuple[int]
        Size of the filter.
        e.g. (256, 256)
    
    cutoff : float
        Cutoff Frequency
        e.g. 0.5

    n : int
        The order [degree] of Chebyshev fitler.
        e.g. 2

    mode : {'low', 'high'}, default: 'low'
        Whether to create a low-pass or high-pass filter

    pos : {'center', 'corner'}, default: 'center'
        The position of the filter.

    norm : bool, default: False
        Normalize the values to be accumulated to 1 [useful in convolution]
        Note: Do not normalize it for frequency multiplication e.g. np.fft.fft2
    
    Returns
    -------
    numpy.ndarray
    """

    rows, cols = size
    x = np.linspace(-1, 1, rows)
    y = np.linspace(-1, 1, cols)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)
    chebyshev_filter = 1 / (1 + (radius / cutoff) ** (2 * n))

    if pos == 'corner':
        chebyshev_filter = np.fft.fftshift(chebyshev_filter)
    elif pos != 'center':
        raise ValueError(f"Invalid `pos` value: {pos}; should be 'center' or 'corner'.")

    if mode == 'high':
        chebyshev_filter = 1 - chebyshev_filter
    elif mode != 'low':
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'low' or 'high'.")

    if norm:
        chebyshev_filter /= np.sum(chebyshev_filter)

    return chebyshev_filter


def bessel(size: tuple[int], cutoff: float, n: int, mode: str = 'low', pos: str = 'center', norm: bool = False) -> np.ndarray:
    """
    Create a 2D Bessel low/high pass filter.

    Parameters
    ----------
    size : tuple[int]
        Size of the filter.
        e.g. (256, 256)
    
    cutoff : float
        Cutoff Frequency
        e.g. 0.5

    n : int
        The order [degree] of Bessel fitler.
        e.g. 2

    mode : {'low', 'high'}, default: 'low'
        Whether to create a low-pass or high-pass filter

    pos : {'center', 'corner'}, default: 'center'
        The position of the filter.

    norm : bool, default: False
        Normalize the values to be accumulated to 1 [useful in convolution]
        Note: Do not normalize it for frequency multiplication e.g. np.fft.fft2
    
    Returns
    -------
    numpy.ndarray
    """

    rows, cols = size
    x = np.linspace(-rows // 2, rows // 2, rows)
    y = np.linspace(-cols // 2, cols // 2, cols)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2) / (rows // 2)
    bessel_filter = np.zeros(size)
    bessel_filter[r <= cutoff] = np.sqrt(1 - (r[r <= cutoff] / cutoff) ** (2 * n))

    if pos == 'corner':
        bessel_filter = np.fft.fftshift(bessel_filter)
    elif pos != 'center':
        raise ValueError(f"Invalid `pos` value: {pos}; should be 'center' or 'corner'.")

    if mode == 'high':
        bessel_filter = 1 - bessel_filter
    elif mode != 'low':
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'low' or 'high'.")

    if norm:
        bessel_filter /= np.sum(bessel_filter)

    return bessel_filter

def triangle(block_size: int, threshold: float= 1.5) -> np.ndarray:
    mask = np.zeros((block_size, block_size))

    for i in range(int(block_size // threshold)):
        mask[i, :int(block_size // threshold) -i - 1] = 1
    
    return mask

def rectangle(block_size: int, threshold: int = 2) -> np.ndarray:
    mask = np.zeros((block_size, block_size))
    mask[:threshold] = 1
    mask[:, :threshold] = 1
    return mask

def block(block_size: int, threshold: int = 2) -> np.ndarray:
    mask = np.zeros((block_size, block_size))
    mask[:threshold, :threshold] = 1
    return mask

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ideal_filter = ideal(size= (256, 256), threshold= 32, shape= 'square', pos= 'center')
    gaussian_filter = gaussian(size= (256, 256), sigma= 32, mode= 'low', pos= 'center', norm= False)
    sinc_filter = sinc(size= (256, 256), sigma= 5, mode= 'high', pos= 'corner')
    butterworth_filter = butterworth(size= (256, 256), cutoff= 32, n= 2, mode= 'low', pos= 'center', norm= False)
    chebyshev_filter = chebyshev(size= (256, 256), cutoff= .5, n= 2, mode= 'low', pos= 'center', norm= False)
    bessel_filter = bessel(size= (256, 256), cutoff= .5, n= 2, mode= 'low', pos= 'center', norm= False)
    triangle_filter = triangle(block_size= 256, threshold= 1.5)
    rectangle_filter = rectangle(block_size= 256, threshold= 32)
    block_filter = block(block_size= 256, threshold= 32)

    fig, axs = plt.subplots(nrows= 3, ncols= 3, layout= 'compressed')
    axs[0, 0].imshow(ideal_filter, cmap= 'gray')
    axs[0, 1].imshow(gaussian_filter, cmap= 'gray')
    axs[0, 2].imshow(sinc_filter, cmap= 'gray')
    axs[1, 0].imshow(butterworth_filter, cmap= 'gray')
    axs[1, 1].imshow(chebyshev_filter, cmap= 'gray')
    axs[1, 2].imshow(bessel_filter, cmap= 'gray')
    axs[2, 0].imshow(triangle_filter, cmap= 'gray')
    axs[2, 1].imshow(rectangle_filter, cmap= 'gray')
    axs[2, 2].imshow(block_filter, cmap= 'gray')

    plt.show()
