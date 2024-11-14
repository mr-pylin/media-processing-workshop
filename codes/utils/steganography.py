import numpy as np

class Steganography:
    """
    Class for implementing steganography techniques.
    
    Attributes:
        encoding (str): Encoding scheme used for the hidden message.
        n_lowest_bits (int): Number of lowest bits to use for hiding the message.
    """
    
    def __init__(self, encoding: str, n_lowest_bits: int) -> None:
        """
        Initialize the Steganography object.
        
        Args:
            encoding (str): Encoding scheme used for the hidden message.
            n_lowest_bits (int): Number of lowest bits to use for hiding the message.
        """
        
        self.encoding = encoding
        self.n_lowest_bits = n_lowest_bits
    
    def encode(self, message: str, image: np.ndarray) -> tuple[int, np.ndarray]:
        """
        Encode a message into an image using steganography.
        
        Args:
            message (str): Message to be encoded.
            image (np.ndarray): Input image to hide the message in.
        
        Returns:
            tuple[int, np.ndarray]: Tuple containing the length of the binary message and the modified image.
        
        Raises:
            ValueError: If the number of lowest bits exceeds the image's bit depth or if the message is too large for the given combination of image and n_lowest_bits.
        """
        
        image_bpp = np.iinfo(image.dtype).bits
        depth = 1 if image.ndim == 2 else len(image[0, 0])
        
        # check that `n_lowest_bits` is valid for the image's bit depth
        if image_bpp < self.n_lowest_bits:
            raise ValueError(f"n_lowest_bits:{self.n_lowest_bits} should be lower or equal to the image's bits per pixel:{image_bpp}")
        
        # check that the image has enough capacity for the message
        if image.shape[0] * image.shape[1] * depth * self.n_lowest_bits < len(message) * image_bpp:
            raise ValueError(f"message is too big for this combination of image & n_lowest_bits:{self.n_lowest_bits} & encoding:{self.encoding}")
        
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
        """
        Decode a hidden message from an image using steganography.
        
        Args:
            image (np.ndarray): Image containing the hidden message.
            length (int): Length of the binary message to decode.
        
        Returns:
            str: Decoded message.
        
        Raises:
            ValueError: If the number of lowest bits exceeds the image's bit depth or if the message length is too large for the given combination of image and n_lowest_bits.
        """
        
        image_bpp = np.iinfo(image.dtype).bits
        depth = 1 if image.ndim == 2 else len(image[0, 0])
        
        # check that `n_lowest_bits` is valid for the image's bit depth
        if image_bpp < self.n_lowest_bits:
            raise ValueError(f"n_lowest_bits:{self.n_lowest_bits} should be lower or equal to the image's bit per pixel:{image_bpp}")
        
        # check that the image has enough capacity for the message
        if image.shape[0] * image.shape[1] * depth *self.n_lowest_bits < length:
            raise ValueError(f"message is too big for this combination of image & n_lowest_bits:{self.n_lowest_bits} & encoding:{self.encoding}")
        
        chunks = []
        image_flat = image.flatten()
        mask = 2 ** self.n_lowest_bits - 1
        num_pixels = (length // self.n_lowest_bits) + (length % self.n_lowest_bits)
        
        for i in range(num_pixels):
            chunks.append(format(image_flat[i] & mask, f"0{self.n_lowest_bits}b"))
        
        binary_message = ''.join(chunk for chunk in chunks)[:length]
        message = ''.join(chr(int(binary_message[i:i + image_bpp], 2)) for i in range(0, len(binary_message), image_bpp))
        
        return message
    
