import numpy as np

# create basis vectors [1d]
def cosine_basis_1d(N: int):
    n = np.arange(N)
    k = n.reshape((N, 1))
    basis_vectors = np.cos((np.pi/N) * k * (n + 0.5))
    return basis_vectors


# create basis images [2d]
def cosine_basis_2d(N: int):
    basis_vectors = cosine_basis_1d(N)

    basis_images = np.zeros(shape= (N, N, N, N))
    for row in range(N):
        for col in range(N):
            basis_images[row, col] = np.outer(basis_vectors[row], basis_vectors[col])

    return basis_images

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