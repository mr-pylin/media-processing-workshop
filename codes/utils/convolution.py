import numpy as np

# a simple function similar to np.pad()
def pad(signal: np.ndarray, pad_width: int, mode: str = 'fill', fill_value: int = 0):
    """
    Pad an array.
    a simple function similar to np.pad()

    Parameters
    ----------
    signal : np.ndarray
        An array

    pad_width : int
        Width of padding in each side

    mode : {'fill', 'circular', 'symmetric'}, default: 'fill'
        'fill': normal padding filled with 'fill_value'
        'circular': circular padding
        'symmetric': symmetric padding

    fillvalue : int, default: 0
        Value to fill padding when 'mode' = 'fill'
    
    Returns
    -------
    numpy.ndarray
    """
    output = np.full(shape= [i + 2 * pad_width for i in signal.shape], fill_value= fill_value)

    # add signal to the center of the output
    if signal.ndim == 1:
        output[pad_width: signal.shape[0] + pad_width] = signal
        if mode == 'circular':
            output[:pad_width], output[-pad_width:] = output[-2 * pad_width: -pad_width], output[pad_width: 2 * pad_width]
        elif mode == 'symmetric':
            output[:pad_width], output[-pad_width:] = output[pad_width: 2 * pad_width][::-1], output[-2 * pad_width: -pad_width][::-1]
        elif mode != 'fill':
            raise ValueError(f"Invalid `mode` value: {mode}; should be 'fill' or 'circular' or 'symmetric'.")
        
    elif signal.ndim == 2:
        output[pad_width: signal.shape[0] + pad_width, pad_width: signal.shape[1] + pad_width] = signal
        if mode == 'circular':
            # Iterate through the rows and columns of the new array
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    # Calculate the corresponding position in the original array
                    # Wrap around the edges if necessary
                    new_i = (i - pad_width) % signal.shape[0]
                    new_j = (j - pad_width) % signal.shape[1]
                    
                    # Assign the value from the original array to the new array
                    output[i, j] = signal[new_i, new_j]
        elif mode == 'symmetric':
            for i in range(pad_width):
                # pad the top and bottom edges
                output[i, :]      = output[2 * pad_width - i - 1, :]
                output[-i - 1, :] = output[pad_width + i + 1    , :]
                
                # pad the left and right edges
                output[:, i]      = output[:, 2 * pad_width - i - 1]
                output[:, -i - 1] = output[:, pad_width + i + 1]

        elif mode != 'fill':
            raise ValueError(f"Invalid `mode` value: {mode}; should be 'fill' or 'circular' or 'symmetric'.")
        
    else:
        raise ValueError(f"Invalid `signal` dim: {signal.ndim}; should be 1D or 2D.")
    
    return output

# a simple function similar to np.convolve()
def convolve_1d(signal: np.ndarray, filter: np.ndarray, mode: str = 'full', boundary: str = 'fill', fill_value: int = 0, correlation: bool = False) -> np.ndarray:
    """
    Convolve two 1-dimensional arrays.
    a simple function similar to np.convolve()

    Parameters
    ----------
    signal : np.ndarray
        An array of length M

    filter : np.ndarray
        An array of length N

    mode : {'full', 'valid', 'same'}, default: 'full'
        'full': The output is the full discrete linear convolution of the inputs (size: M+N-1)
        'valid': The output consists only of those elements that do not rely on the zero-padding (size: M-N+1)
        'same': The output is the same size as signal (size: M)

    boundary : {'fill', 'circular', 'symmetric'}, default: 'fill'
        Shape of the filter.

    fill_value : int, default: 0
        Value to fill padding when 'boundary' = 'fill'

    correlation : bool, default: False
        To calculate correlation instead of cconvolution
        
    Returns
    -------
    numpy.ndarray
    """

    assert signal.ndim == 1 and filter.ndim == 1, "signal & filter must be 1D arrays"

    # mode
    if mode == 'full':
        pad_width = filter.shape[0] - 1
        output_length = signal.shape[0] + filter.shape[0] - 1
    elif mode == 'valid':
        pad_width = 0
        output_length = signal.shape[0] - filter.shape[0] + 1
    elif mode == 'same':
        pad_width = (filter.shape[0] - 1) // 2
        output_length = signal.shape[0]
    else:
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'fill' or 'circular' or 'symmetric'.")

    # padding signal if needed
    padded_signal = pad(signal, pad_width= pad_width, mode= boundary, fill_value= fill_value)

    # reverse of filter
    if not correlation:
        filter_reversed = filter[::-1]

    # convolution
    output = np.empty(shape= output_length)

    for i in range(output_length):
        output[i] = np.dot(padded_signal[i: i + filter.shape[0]], filter_reversed)

    return output

# a simple function similar to scipy.signal.convolve2d
def convolve_2d(signal: np.ndarray, filter: np.ndarray, mode: str = 'full', boundary: str = 'fill', fill_value: int = 0, correlation: bool = False) -> np.ndarray:
    """
    Convolve two 2-dimensional arrays.
    a simple function similar to scipy.signal.convolve2d

    Parameters
    ----------
    signal : np.ndarray
        An array of length M

    filter : np.ndarray
        An array of length N

    mode : {'full', 'valid', 'same'}, default: 'full'
        'full': The output is the full discrete linear convolution of the inputs (size: M+N-1)
        'valid': The output consists only of those elements that do not rely on the zero-padding (size: M-N+1)
        'same': The output is the same size as signal (size: M)

    boundary : {'fill', 'circular', 'symmetric'}, default: 'fill'
        Shape of the filter.

    fill_value : int, default: 0
        Value to fill padding when 'boundary' = 'fill'

    correlation : bool, default: False
        To calculate correlation instead of cconvolution

    Returns
    -------
    numpy.ndarray
    """

    assert signal.ndim == 2 and filter.ndim == 2, "signal & filter must be 2D arrays"

    # mode
    if mode == 'full':
        pad_width = filter.shape[0] - 1
        output_length = (signal.shape[0] + filter.shape[0] - 1, signal.shape[1] + filter.shape[0] - 1)
    elif mode == 'valid':
        pad_width = 0
        output_length = (signal.shape[0] - filter.shape[0] + 1, signal.shape[1] - filter.shape[0] + 1)
    elif mode == 'same':
        pad_width = (filter.shape[0] - 1) // 2
        output_length = signal.shape
    else:
        raise ValueError(f"Invalid `mode` value: {mode}; should be 'fill' or 'circular' or 'symmetric'.")

    # padding signal if needed
    padded_signal = pad(signal, pad_width= pad_width, mode= boundary, fill_value= fill_value)

    # reverse of filter
    if not correlation:
        filter_reversed = np.flip(np.flip(filter, axis= 1), axis= 0)

    # convolution
    output = np.empty(shape= output_length)

    for row in range(output_length[0]):
        for col in range(output_length[1]):
            output[row, col] = np.multiply(padded_signal[row: row + filter.shape[0], col: col + filter.shape[1]], filter_reversed).sum()

    return output

if __name__ == '__main__':
    from scipy.signal import convolve2d

    arr1 = np.arange(5)
    filter_1 = np.array([1, 2, 3])

    arr2 = np.arange(25).reshape(5, 5)
    filter_2 = np.arange(9).reshape(3, 3)

    print(pad(signal= arr1, pad_width= 2, mode= 'fill', fill_value= 2))
    print(pad(signal= arr1, pad_width= 2, mode= 'circular'))
    print(pad(signal= arr1, pad_width= 2, mode= 'symmetric'))

    print(pad(signal= arr2, pad_width= 2, mode= 'fill', fill_value= 0))
    print(pad(signal= arr2, pad_width= 2, mode= 'circular'))
    print(pad(signal= arr2, pad_width= 2, mode= 'symmetric'))

    print(convolve_1d(arr1, filter_1, mode= 'same', boundary= 'fill', fill_value= 0))
    print(convolve_2d(arr2, filter_2, mode= 'valid', boundary= 'fill', fill_value= 0))