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
    Create a 2D circled Gaussian low/high pass filter.

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

    # create a centered circled gaussian low-pass filter
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

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ideal_filter = ideal(size= (256, 256), threshold= 32, shape= 'square', pos= 'center')
    gaussian_filter = gaussian(size= (256, 256), sigma= 32, mode= 'low', pos= 'center', norm= False)
    sinc_filter = sinc(size= (256, 256), sigma= 5, mode= 'high', pos= 'corner')

    fig, axs = plt.subplots(nrows= 1, ncols= 3)
    axs[0].imshow(ideal_filter, cmap= 'gray')
    axs[1].imshow(gaussian_filter, cmap= 'gray')
    axs[2].imshow(sinc_filter, cmap= 'gray')
    plt.show()
