import numpy as np
from typing import Sequence

# create basis vectors [1-dimensional]
def cosine_basis_1d(N: int) -> np.ndarray:
    """
    Generate a set of 1D cosine basis vectors.
    
    Args:
        N (int): The number of basis vectors to generate.
    
    Returns:
        np.ndarray: A 2D array of shape (N, N) containing the cosine basis vectors.
    """
    
    n = np.arange(N)
    k = n.reshape((N, 1))
    basis_vectors = np.cos((np.pi/N) * k * (n + 0.5))
    return basis_vectors

# create basis images [2-dimensional] using <cosine_basis_1d>
def cosine_basis_2d(N: int) -> np.ndarray:
    """
    Generate a set of 2D cosine basis images using cosine_basis_1d.
    
    Args:
        N (int): The size of the 2D basis image (N x N).
    
    Returns:
        np.ndarray: A 4D array of shape (N, N, N, N) containing the 2D cosine basis images.
    """
    
    basis_vectors = cosine_basis_1d(N)
    
    # basis_images = np.zeros(shape= (N, N, N, N))
    # for row in range(N):
    #     for col in range(N):
    #         basis_images[row, col] = np.outer(basis_vectors[row], basis_vectors[col])
    
    # this is equivalent to above commented codes [optimized version]
    reshaped_basis_vectors_row = basis_vectors[:, np.newaxis, :, np.newaxis] # shape: 2x1x2x1
    reshaped_basis_vectors_col = basis_vectors[np.newaxis, :, np.newaxis, :] # shape: 2x1x2x1
    basis_images = np.matmul(reshaped_basis_vectors_row, reshaped_basis_vectors_col)
    
    return basis_images

# discrete Cosine Transform [1-dimensional]
def dct(signal: Sequence) -> np.ndarray:
    """
    Perform a discrete cosine transform (DCT) on a 1D signal.
    
    Args:
        signal (Sequence): The 1D input signal to be transformed.
    
    Returns:
        np.ndarray: The frequency domain representation of the signal after DCT.
    """
    
    N = len(signal)
    basis_vectors = cosine_basis_1d(N)
    frequency_domain = np.dot(basis_vectors, signal)
    
    # scaling factor for the DC value is actually np.sqrt(1/N) so divide it by np.sqrt(2)
    frequency_domain *= np.sqrt(2 / N)
    frequency_domain[0] /= np.sqrt(2)
    
    return frequency_domain

# inverse discrete Cosine Transform [1-dimensional]
# in Cosine, backward-transform is just the transposed version of forward-transform
def idct(signal: Sequence) -> np.ndarray:
    """
    Perform an inverse discrete cosine transform (IDCT) on a 1D signal.
    
    Args:
        signal (Sequence): The 1D frequency domain signal to be transformed.
    
    Returns:
        np.ndarray: The spatial domain representation of the signal after IDCT.
    """
    
    N = len(signal)
    basis_vectors = cosine_basis_1d(N)
    
    # scaling factor for the DC value is actually np.sqrt(1/N) so divide it by np.sqrt(2)
    signal *= np.sqrt(2 / N)
    signal[0] /= np.sqrt(2)
    
    spatial_domain = np.dot(basis_vectors.T, signal)
    
    return spatial_domain

# discrete Cosine Transform [2-dimensional]
def dct2(signal: Sequence) -> np.ndarray:
    """
    Perform a 2D discrete cosine transform (DCT) on a squared image.
    
    Args:
        signal (Sequence): The 2D input signal (image) to be transformed.
    
    Raises:
        AssertionError: If the input signal is not a square image.
    
    Returns:
        np.ndarray: The 2D frequency domain representation of the image after DCT.
    """
    
    M, N = signal.shape
    assert M == N, "This function can only handle squared images"
    
    basis_images = cosine_basis_2d(M)
    
    # frequency_domain = np.zeros(shape= signal.shape, dtype= np.float64)
    # for row in range(M):
    #     for col in range(N):
    #         frequency_domain[row, col] = np.multiply(basis_images[row, col], signal).sum()
    
    # this is equivalent to above commented codes [optimized version]
    frequency_domain = np.tensordot(basis_images, signal, axes=([2, 3], [0, 1])) * np.sqrt(4 / (N * M))
    
    # scaling factor for the first column & row values is actually np.sqrt(1/(N*M)) so divide it by np.sqrt(4)
    frequency_domain[0, :] /= np.sqrt(2)
    frequency_domain[:, 0] /= np.sqrt(2)
    
    return frequency_domain

# inverse discrete Cosine Transform [2-dimensional]
def idct2(signal: Sequence):
    """
    Perform an inverse 2D discrete cosine transform (IDCT) on a squared image.
    
    Args:
        signal (Sequence): The 2D frequency domain signal (image) to be transformed.
    
    Raises:
        AssertionError: If the input signal is not a square image.
    
    Returns:
        np.ndarray: The 2D spatial domain representation of the image after IDCT.
    """
    
    M, N = signal.shape
    assert M == N, "This function can only handle squared images"
    
    basis_images = np.transpose(cosine_basis_2d(M))
    
    signal *= np.sqrt(4 / (N * M))
    
    # scaling factor for the first column & row values is actually np.sqrt(1/(N*M)) so divide it by np.sqrt(4)
    signal[0, :] /= np.sqrt(2)
    signal[:, 0] /= np.sqrt(2)
    # spatial_domain = np.zeros(shape= signal.shape, dtype= np.float64)
    # for row in range(M):
    #     for col in range(N):
    #         spatial_domain[row, col] = np.multiply(basis_images[row, col], signal).sum()
    
    # this is equivalent to above commented codes [optimized version]
    spatial_domain = np.tensordot(basis_images, signal, axes=([2, 3], [0, 1]))
    
    return spatial_domain