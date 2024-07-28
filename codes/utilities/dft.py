import numpy as np
from typing import Sequence

# create basis vectors [1-dimensional]
def fourier_basis_1d(N: int) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape((N, 1))

    # basis_vectors = np.zeros((N, N), dtype=np.complex128)
    # for k in range(N):
    #     for n in range(N):
    #         basis_vectors[k, n] = np.exp(-2j * np.pi * k * n / N)

    # this is equivalent to above commented codes [optimized version]
    basis_vectors = np.exp(-2j * np.pi * k * n / N)

    return basis_vectors

# create basis images [2-dimensional] using <fourier_basis_1d>
def fourier_basis_2d(N: int):
    basis_vectors = fourier_basis_1d(N)

    # basis_images = np.zeros(shape= (N, N, N, N), dtype= np.complex128)
    # for row in range(N):
    #     for col in range(N):
    #         basis_images[row, col] = np.outer(basis_vectors[row], basis_vectors[col])

    # this is equivalent to above commented codes [optimized version]
    reshaped_basis_vectors_row = basis_vectors[:, np.newaxis, :, np.newaxis] # shape: 2x1x2x1
    reshaped_basis_vectors_col = basis_vectors[np.newaxis, :, np.newaxis, :] # shape: 2x1x2x1
    basis_images = np.matmul(reshaped_basis_vectors_row, reshaped_basis_vectors_col)

    return basis_images

# discrete Fourier Transform [1-dimensional]
def dft(signal: Sequence, norm: str = 'backward'):
    N = len(signal)
    basis_vectors = fourier_basis_1d(N)
    frequency_domain = np.dot(basis_vectors, signal)

    # scale all values with respect to the <norm>
    if norm == 'forward':
        frequency_domain *= (1 / N)
    elif norm == 'ortho':
        frequency_domain *= np.sqrt(1 / N)
    elif norm != 'backward':
        raise ValueError(f"Invalid `norm` value: {norm}; Allowed values: {{'forward', 'ortho', 'backward'}}.")

    return frequency_domain

# inverse discrete Fourier Transform [1-dimensional]
# in fourier, backward-transform is just the conjugate transposed version of forward-transform
def idft(signal: Sequence, norm: str = 'backward'):
    N = len(signal)
    basis_vectors = fourier_basis_1d(N)
    spatial_domain = np.dot(np.conj(basis_vectors.T), signal)

    # scale all values with respect to the <norm>
    if norm == 'ortho':
        spatial_domain *= np.sqrt(1 / N)
    elif norm == 'backward':
        spatial_domain *= (1 / N)
    elif norm != 'forward':
        raise ValueError(f"Invalid `norm` value: {norm}; Allowed values: {{'forward', 'ortho', 'backward'}}.")

    return spatial_domain

# discrete Fourier Transform [2-dimensional]
def dft2(signal: Sequence, norm: str = 'backward') -> np.ndarray:
    M, N = signal.shape
    assert M == N, "This function can only handle squared images"

    basis_images = fourier_basis_2d(M)

    # frequency_domain = np.zeros(shape= signal.shape, dtype= np.complex128)
    # for row in range(M):
    #     for col in range(N):
    #         frequency_domain[row, col] = np.multiply(basis_images[row, col], signal).sum()

    # this is equivalent to above commented codes [optimized version]
    frequency_domain = np.tensordot(basis_images, signal, axes=([2, 3], [0, 1]))

    # scale all values with respect to the <norm>
    if norm == 'forward':
        frequency_domain *= (1 / (N * M))
    elif norm == 'ortho':
        frequency_domain *= np.sqrt(1 / (N * M))
    elif norm != 'backward':
        raise ValueError(f"Invalid `norm` value: {norm}; Allowed values: {{'forward', 'ortho', 'backward'}}.")

    return frequency_domain

# inverse discrete Fourier Transform [2-dimensional]
def idft2(signal: Sequence, norm: str = 'backward'):
    M, N = signal.shape
    assert M == N, "This function can only handle squared images"
    
    basis_images = np.conjugate(fourier_basis_2d(M))

    # spatial_domain = np.zeros(shape= signal.shape, dtype= np.complex128)
    # for row in range(M):
    #     for col in range(N):
    #         spatial_domain[row, col] = np.multiply(basis_images[row, col], signal).sum()

    # this is equivalent to above commented codes [optimized version]
    spatial_domain = np.tensordot(basis_images, signal, axes=([2, 3], [0, 1]))

    # scale all values with respect to the <norm>
    if norm == 'ortho':
        spatial_domain *= np.sqrt(1 / (N * M))
    elif norm == 'backward':
        spatial_domain *= (1 / (N * M))
    elif norm != 'forward':
        raise ValueError(f"Invalid `norm` value: {norm}; Allowed values: {{'forward', 'ortho', 'backward'}}.")

    return spatial_domain

# fftshift : shift low-frequencies to the center
def fftshift(arr):
    N = arr.shape[0]
    mid = N // 2
    shifted_arr = np.empty_like(arr)
    
    shifted_arr[:mid, :mid] = arr[mid:, mid:]
    shifted_arr[mid:, mid:] = arr[:mid, :mid]
    shifted_arr[:mid, mid:] = arr[mid:, :mid]
    shifted_arr[mid:, :mid] = arr[:mid, mid:]
    
    return shifted_arr

# ifftshift : shift low-frequencies to the corners
def ifftshift(arr):
    N = arr.shape[0]
    mid = N // 2
    shifted_arr = np.empty_like(arr)
    
    shifted_arr[mid:, mid:] = arr[:mid, :mid]
    shifted_arr[:mid, :mid] = arr[mid:, mid:]
    shifted_arr[mid:, :mid] = arr[:mid, mid:]
    shifted_arr[:mid, mid:] = arr[mid:, :mid]
    
    return shifted_arr

if __name__ == '__main__':
    print(fourier_basis_1d(5).shape)
    print(fourier_basis_2d(5).shape)

    arr_1      = np.arange(3)
    dft_arr_1  = dft(arr_1)
    idft_arr_1 = np.round(idft(dft_arr_1).real)
    print(arr_1)
    print(dft_arr_1)
    print(idft_arr_1)

    arr_2      = np.arange(4).reshape(2, 2)
    dft_arr_2  = dft2(arr_2)
    idft_arr_2 = np.round(idft2(dft_arr_2).real)
    print(arr_2)
    print(dft_arr_2)
    print(idft_arr_2)