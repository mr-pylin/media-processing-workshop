import numpy as np
from typing import Sequence

# create basis vectors [1d]
def fourier_basis_1d(N: int):
    n = np.arange(N)
    k = n.reshape((N, 1))
    basis_vectors = np.exp(-2j * np.pi * k * n / N)

    # another method which is inefficient because of looping:

    # basis_vectors = np.zeros((N, N), dtype=np.complex128)
    # for k in range(N):
    #     for n in range(N):
    #         basis_vectors[k, n] = np.exp(-2j * np.pi * k * n / N)

    return basis_vectors

# 1-dimensional discrete Fourier Transform
# the whole constant multiplier is applied in the inverse function
def dft(signal: Sequence):
    N = len(signal)
    basis_vectors = fourier_basis_1d(N)
    frequency_domain = np.dot(basis_vectors, signal)
    return frequency_domain

# 1-dimensional inverse discrete Fourier Transform
# in fourier, backward-transform is just the conjugate transposed version of forward-transform
def idft(frequency_domain: Sequence):
    N = len(frequency_domain)
    basis_vectors = fourier_basis_1d(N)
    reconstructed_signal = np.dot(np.conj(basis_vectors.T), frequency_domain) / N
    return reconstructed_signal

# create basis images [2d]
def fourier_basis_2d(N: int):
    basis_vectors = fourier_basis_1d(N)

    basis_images = np.zeros(shape= (N, N, N, N), dtype= np.complex128)
    for row in range(N):
        for col in range(N):
            basis_images[row, col] = np.outer(basis_vectors[row], basis_vectors[col])

    return basis_images

# 2-dimensional discrete Fourier Transform
def dft2(signal: Sequence) -> np.ndarray:
    M, N = signal.shape
    assert M == N, "This function can only handle squared images"

    basis_images = fourier_basis_2d(M)

    frequency_domain = np.zeros(shape= signal.shape, dtype= np.complex128)
    for row in range(M):
        for col in range(N):
            frequency_domain[row, col] = np.multiply(basis_images[row, col], signal).sum()

    return frequency_domain

# 2-dimensional inverse discrete Fourier Transform
def idft2(frequency_domain: Sequence):
    M, N = frequency_domain.shape
    assert M == N, "This function can only handle squared images"
    
    basis_images = np.conjugate(fourier_basis_2d(M)) / (M * N)

    spatial_domain = np.zeros(shape= frequency_domain.shape, dtype= np.complex128)
    for row in range(M):
        for col in range(N):
            spatial_domain[row, col] = np.multiply(basis_images[row, col], frequency_domain).sum()

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