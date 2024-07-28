import numpy as np

class Steganography:

    def __init__(self, encoding: str, n_lowest_bits: int) -> None:
        self.encoding = encoding
        self.n_lowest_bits = n_lowest_bits

    def encode(self, message: str, image: np.ndarray) -> tuple[int, np.ndarray]:
        image_bpp = np.iinfo(image.dtype).bits
        depth = 1 if image.ndim == 2 else len(image[0, 0])

        assert image_bpp >= self.n_lowest_bits, f"n_lowest_bits:{self.n_lowest_bits} should be lower or equal to the image's bit per pixel:{image_bpp}"
        assert image.shape[0] * image.shape[1] * depth * self.n_lowest_bits >= len(message) * image_bpp, f"message is too big for this combination of image & n_lowest_bits:{self.n_lowest_bits} & encoding:{self.encoding}"

        mask = 2 ** image_bpp - 2 ** self.n_lowest_bits
        binary_message = ''.join(format(ord(char), '08b') for char in message)
        image_flat = image.flatten()

        message_length = len(binary_message)
        counter = 0

        while message_length >= 0:

            image_flat[counter] = (
                (image_flat[counter] & mask) | int(format(binary_message[counter * self.n_lowest_bits: counter * self.n_lowest_bits + self.n_lowest_bits], f"0<{self.n_lowest_bits}"), 2)
            )

            counter += 1
            message_length -= self.n_lowest_bits

        return len(binary_message), image_flat.reshape(image.shape)

    def decode(self, image: np.ndarray, length: int) -> str:
        image_bpp = np.iinfo(image.dtype).bits
        depth = 1 if image.ndim == 2 else len(image[0, 0])

        assert image_bpp >= self.n_lowest_bits, f"n_lowest_bits:{self.n_lowest_bits} should be lower or equal to the image's bit per pixel:{image_bpp}"
        assert image.shape[0] * image.shape[1] * depth *self.n_lowest_bits >= length, f"message is too big for this combination of image & n_lowest_bits:{self.n_lowest_bits} & encoding:{self.encoding}"

        chunks = []
        image_flat = image.flatten()
        mask = 2 ** self.n_lowest_bits - 1
        num_pixels = (length // self.n_lowest_bits) + (length % self.n_lowest_bits)

        for i in range(num_pixels):
            chunks.append(format(image_flat[i] & mask, f"0{self.n_lowest_bits}b"))

        binary_message = ''.join(chunk for chunk in chunks)[:length]
        message = ''.join(chr(int(binary_message[i:i + image_bpp], 2)) for i in range(0, len(binary_message), image_bpp))

        return message
    
