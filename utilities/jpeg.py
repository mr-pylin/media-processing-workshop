import numpy as np
from scipy import fftpack
from numpy import typing as npt

class JPEG:

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

    def encode(self) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:

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

    def decode(
            self,
            dc: npt.NDArray[np.float32],
            ac: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.uint8]:

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
                if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
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