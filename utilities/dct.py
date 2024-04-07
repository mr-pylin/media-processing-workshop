import numpy as np
from typing import Sequence

# create basis vectors [1d]
def cosine_basis_1d(N: int):
    n = np.arange(N)
    k = n.reshape((N, 1))
    basis_vectors = np.cos((np.pi/N) * k * (n + 0.5))
    return basis_vectors

# 1-dimensional discrete Cosine Transform
# the whole constant multiplier is applied in the inverse function
def dct(signal: Sequence):
    N = len(signal)
    basis_vectors = cosine_basis_1d(N)
    frequency_domain = np.dot(basis_vectors, signal)# * np.sqrt(2/N)

    frequency_domain[0] /= np.sqrt(2)
    
    return frequency_domain

# 1-dimensional inverse discrete Cosine Transform
# in Cosine, backward-transform is just the transposed version of forward-transform
def idct(frequency_domain: Sequence):
    N = len(frequency_domain)
    basis_vectors = cosine_basis_1d(N)
    reconstructed_signal = np.dot(basis_vectors.T, frequency_domain) * (2/N) # instead of * np.sqrt(2/N)
    return reconstructed_signal

# create basis images [2d]
def cosine_basis_2d(N: int):
    basis_vectors = cosine_basis_1d(N)

    basis_images = np.zeros(shape= (N, N, N, N))
    for row in range(N):
        for col in range(N):
            basis_images[row, col] = np.outer(basis_vectors[row], basis_vectors[col])

    return basis_images

# 2-dimensional discrete Cosine Transform
def dct2(signal: Sequence) -> np.ndarray:
    M, N = signal.shape
    assert M == N, "This function can only handle squared images"

    basis_images = cosine_basis_2d(M)

    # frequency_domain = np.zeros(shape= signal.shape, dtype= np.float64)
    # for row in range(M):
    #     for col in range(N):
    #         frequency_domain[row, col] = np.multiply(basis_images[row, col], signal).sum()

    # this is equivalent to above commented codes
    frequency_domain = np.tensordot(basis_images, signal, axes=([2, 3], [0, 1]))
    
    frequency_domain[0, :] /= np.sqrt(4)
    frequency_domain[:, 0] /= np.sqrt(4)

    return frequency_domain

# 2-dimensional inverse discrete Cosine Transform
def idct2(frequency_domain: Sequence):
    M, N = frequency_domain.shape
    assert M == N, "This function can only handle squared images"
    
    basis_images = np.transpose(cosine_basis_2d(M)) * (2/N) * (2/M)

    # spatial_domain = np.zeros(shape= frequency_domain.shape, dtype= np.float64)
    # for row in range(M):
    #     for col in range(N):
    #         spatial_domain[row, col] = np.multiply(basis_images[row, col], frequency_domain).sum()

    # this is equivalent to above commented codes
    spatial_domain = np.tensordot(basis_images, frequency_domain, axes=([2, 3], [0, 1]))

    return spatial_domain

# masks
def triangle_mask(block_size: int, threshold: float= 1.5) -> np.ndarray:
    mask = np.zeros((block_size, block_size))

    for i in range(int(block_size // threshold)):
        mask[i, :int(block_size // threshold) -i - 1] = 1
    
    return mask

def rectangle_mask(block_size: int, threshold: int = 2) -> np.ndarray:
    mask = np.zeros((block_size, block_size))
    mask[:threshold] = 1
    mask[:, :threshold] = 1
    return mask

def block_mask(block_size: int, threshold: int = 2) -> np.ndarray:
    mask = np.zeros((block_size, block_size))
    mask[:threshold, :threshold] = 1
    return mask