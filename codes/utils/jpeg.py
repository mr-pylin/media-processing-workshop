import numpy as np
from scipy import fftpack

# this implementation may not be an exact replica of JPEG, but it captures the core concept.
class JPEG:
    """
    Class for implementing JPEG compression and decompression.
    
    Attributes:
        q_table (np.ndarray): Quantization table used for JPEG compression.
        image (np.ndarray): Input image to be compressed.
        scale (int): Scaling factor for the quantization table.
        block_size (int): Size of the blocks used in JPEG compression.
        height (int): Height of the input image.
        width (int): Width of the input image.
        depth (int): Depth of the input image (1 for grayscale, 3 for RGB).
        num_row_blocks (int): Number of blocks along the rows.
        num_col_blocks (int): Number of blocks along the columns.
    """
    
    q_table = np.array([
        [16, 11, 10, 16,  24,  40,  51,  60],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99],
    ])
    
    def __init__(self, image: np.ndarray, scale: int = 1) -> None:
        """
        Initialize the JPEG object.
        
        Args:
            image (np.ndarray): Input image to be compressed.
            scale (int, optional): Scaling factor for the quantization table. Defaults to 1.
        
        Raises:
            AssertionError: If the input image is not grayscale or RGB, or if its dimensions are not multiples of the block size.
        """
        
        assert len(image.shape) in [2, 3], f"only GrayScale/RGB image is supported. current image has shape: {image.shape}"
        if image.ndim == 3:
            assert image.shape[2] == 3, f"the depth of the image should be 3. current depth is: {image.shape[2]}"
        
        self.image = image
        self.scale = scale
        self.block_size = 8
        self.height, self.width = image.shape[:2]
        
        if image.ndim == 3:
            self.depth = image.shape[2]
        else:
            self.depth = 1
        
        assert self.height % self.block_size == 0, f"video height {self.height} is not divisible to {self.block_size}"
        assert self.width % self.block_size == 0, f"video width {self.width} is not divisible to {self.block_size}"
        
        self.num_row_blocks = self.height // self.block_size
        self.num_col_blocks = self.width // self.block_size
    
    def encode(self) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
        """
        Encode the input image using JPEG compression.
        
        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing DC coefficients and AC coefficients.
        """
        
        # np.uint8 -> np.float32
        image = self.image.astype(np.float32)
        
        # shift intensity [0, 255] -> [-128, 127]
        image -= 128
        
        # block the frame
        blocks  = image.reshape(
            self.num_row_blocks,
            self.block_size,
            self.num_col_blocks,
            self.block_size,
            self.depth
        ).swapaxes(1, 2)
        
        # 2D DCT + quantization section
        dc = np.zeros(shape= (self.num_row_blocks * self.num_col_blocks, self.depth), dtype= np.float32)
        ac = np.zeros(shape= (self.num_row_blocks, self.num_col_blocks, self.block_size ** 2 - 1, self.depth), dtype= np.float32)
        
        # loop over blocks
        for row in range(self.num_row_blocks):
            for col in range(self.num_col_blocks):
                for channel in range(self.depth):
                    
                    # 2D DCT
                    dct = fftpack.dctn(blocks[row, col, :, :, channel], norm= 'ortho')
                    
                    # quantization
                    quantize = np.round(dct / (JPEG.q_table * self.scale))
                    
                    # zigzag order
                    zigzag = self.__zigzag(quantize)
                    dc[row * self.num_col_blocks + col, channel] = zigzag[0]
                    ac[row, col, :, channel] = zigzag[1:]
        
        return (dc, ac)
    
    def decode(self, dc: np.ndarray[np.float32], ac: np.ndarray[np.float32]) -> np.ndarray[np.uint8]:
        """
        Decode the compressed data back to an image.
        
        Args:
            dc (np.ndarray): DC coefficients.
            ac (np.ndarray): AC coefficients.
        
        Returns:
            np.ndarray: Reconstructed image.
        """
        
        # dequantization + 2D IDCT section
        image  = np.zeros(shape= (self.num_row_blocks, self.num_col_blocks, self.block_size, self.block_size, self.depth), dtype= np.float32)
        
        # loop over blocks
        for row in range(self.num_row_blocks):
            for col in range(self.num_col_blocks):
                for channel in range(self.depth):
                    
                    # dc+ac -> block
                    stream = np.hstack((dc[row * self.num_col_blocks + col, channel], ac[row, col, :, channel]))
                    
                    # inverse zigzag order
                    block = self.__inverse_zigzag(stream)
                    
                    # dequantization
                    dequantize = block  * (JPEG.q_table * self.scale)
                    
                    # 2D IDCT
                    image[row, col, :, :, channel] = fftpack.idctn(dequantize, norm= 'ortho')
        
        # unblock the frame
        image  = image.swapaxes(1, 2).reshape(self.height, self.width, self.depth)
        
        # shift intensity [-128, 127] -> [0, 255]
        image += 128
        
        # clip range to [0, 255] & change dtype to np.uint8
        reconstructed_image = image.clip(0, 255).astype(np.uint8)
        
        return reconstructed_image.squeeze()
    
    def __zigzag(self, image: np.ndarray) -> np.ndarray:
        """
        Perform zigzag ordering on a 2D array.
        
        Args:
            image (np.ndarray): 2D array to be reordered.
        
        Returns:
            np.ndarray: 1D array representing the zigzag ordered elements of the input.
        """
        
        h = 0
        v = 0
        
        vmin = 0
        hmin = 0
        
        vmax = image.shape[0]
        hmax = image.shape[1]
        
        i = 0
        
        output = np.zeros((vmax * hmax), dtype= np.float32)
        while ((v < vmax) and (h < hmax)):
            if ((h + v) % 2) == 0:                      # going up
                if (v == vmin):
                    output[i] = image[v, h]             # if we got to the first line
                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        
                    i = i + 1
                elif ((h == hmax -1 ) and (v < vmax)):  # if we got to the last column
                    output[i] = image[v, h] 
                    v = v + 1
                    i = i + 1
                elif ((v > vmin) and (h < hmax -1 )):   # all other cases
                    output[i] = image[v, h] 
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:                                       # going down
                if ((v == vmax -1) and (h <= hmax -1)): # if we got to the last line
                    output[i] = image[v, h] 
                    h = h + 1
                    i = i + 1
                elif (h == hmin):                       # if we got to the first column
                    output[i] = image[v, h] 
                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                elif ((v < vmax -1) and (h > hmin)):    # all other cases
                    output[i] = image[v, h] 
                    v = v + 1
                    h = h - 1
                    i = i + 1
            if ((v == vmax-1) and (h == hmax-1)):       # bottom right element
                output[i] = image[v, h] 
                break
        return output
    
    def __inverse_zigzag(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inverse zigzag ordering on a 1D array.
        
        Args:
            image (np.ndarray): 1D array to be reordered.
        
        Returns:
            np.ndarray: 2D array representing the inverse zigzag ordered elements of the input.
        """
        
        h = 0
        v = 0
        
        vmin = 0
        hmin = 0
        
        vmax = self.block_size
        hmax = self.block_size
        
        output = np.zeros((vmax, hmax), dtype= np.float32)
        
        i = 0
        
        while ((v < vmax) and (h < hmax)): 
                                                        # going up
            if ((h + v) % 2) == 0:                 
                if (v == vmin):
                    output[v, h] = image[i]             # if we got to the first line
                    if (h == hmax):
                        v = v + 1
                    else:
                        h = h + 1                        
                    i = i + 1
                elif ((h == hmax -1 ) and (v < vmax)):  # if we got to the last column
                    output[v, h] = image[i] 
                    v = v + 1
                    i = i + 1
                elif ((v > vmin) and (h < hmax -1 )):   # all other cases
                    output[v, h] = image[i] 
                    v = v - 1
                    h = h + 1
                    i = i + 1
            else:                                       # going down
                if ((v == vmax -1) and (h <= hmax -1)): # if we got to the last line
                    output[v, h] = image[i] 
                    h = h + 1
                    i = i + 1
                elif (h == hmin):                       # if we got to the first column
                    output[v, h] = image[i] 
                    if (v == vmax -1):
                        h = h + 1
                    else:
                        v = v + 1
                    i = i + 1
                elif((v < vmax -1) and (h > hmin)):     # all other cases
                    output[v, h] = image[i] 
                    v = v + 1
                    h = h - 1
                    i = i + 1
            if ((v == vmax-1) and (h == hmax-1)):       # bottom right element
                output[v, h] = image[i] 
                break
        return output