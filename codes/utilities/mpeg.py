# note: this implementation may not be an exact replica of MPEG2, but it captures the core concept.

import numpy as np
from scipy import fftpack
import cv2
from numpy import typing as npt

class MPEG:

    q_luminance_table = np.array([
        [16, 11, 10, 16,  24,  40,  51,  60],
        [12, 12, 14, 19,  26,  58,  60,  55],
        [14, 13, 16, 24,  40,  57,  69,  56],
        [14, 17, 22, 29,  51,  87,  80,  62],
        [18, 22, 37, 56,  68, 109, 103,  77],
        [24, 35, 55, 64,  81, 104, 113,  92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103,  99],
    ])

    q_chrominance_table = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99 ,99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ])

    def __init__(self, video: np.ndarray, scale: int = 1, search_area: int = 4) -> None:

        assert len(video.shape) == 4, f"only RGB video is supported. current video has shape: {video.shape}"

        self.actual_frames = video
        self.reconstructed_frames = np.zeros_like(video)

        self.scale = scale
        self.search_area = search_area
        self.macroblock_size = 16
        self.block_size = 8

        self.frames, self.height, self.width, self.depth = video.shape
        assert self.height % self.macroblock_size == 0, f"video height {self.height} is not divisible to {self.macroblock_size}"
        assert self.width % self.macroblock_size == 0, f"video width {self.width} is not divisible to {self.macroblock_size}"

        self.num_row_blocks = self.height // self.block_size
        self.num_col_blocks = self.width // self.block_size

        self.num_row_macroblocks = self.num_row_blocks // 2
        self.num_col_macroblocks = self.num_col_blocks // 2

    def i_encode(
            self,
            frame: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:

        # np.uint8 -> np.float32
        frame = frame.astype(np.float32)

        # convert RGB to YCrCb
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        # split channels
        y, cr, cb = cv2.split(ycrcb)

        # downsample chrominance channels [4:2:0]
        cr = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2))
        cb = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2))

        # shift intensity [0, 255] -> [-128, 127]
        y  -= 128
        cr -= 128
        cb -= 128

        # block the frame
        y_blocks  = y.reshape(y.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
        cr_blocks = cr.reshape(cr.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
        cb_blocks = cb.reshape(cb.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)

        # 2D DCT + quantization section
        y_dct  = np.zeros_like(y_blocks)
        cr_dct = np.zeros_like(cr_blocks)
        cb_dct = np.zeros_like(cb_blocks)

        # loop over blocks
        for row in range(self.num_row_blocks):
            for col in range(self.num_col_blocks):

                # 2D DCT + quantization
                y_dct[row, col] = np.round(fftpack.dctn(y_blocks[row, col], norm= 'ortho') / (MPEG.q_luminance_table * self.scale))

                if row < self.num_row_blocks // 2 and col < self.num_col_blocks // 2: # chrominance channels are downsampled
                    cr_dct[row, col] = np.round(fftpack.dctn(cr_blocks[row, col], norm= 'ortho') / (MPEG.q_chrominance_table * self.scale))
                    cb_dct[row, col] = np.round(fftpack.dctn(cb_blocks[row, col], norm= 'ortho') / (MPEG.q_chrominance_table * self.scale))

        # unblock the frame
        y_dct  = y_dct.swapaxes(1, 2).reshape(self.height, self.width)
        cr_dct = cr_dct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)
        cb_dct = cb_dct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)

        return (y_dct, cr_dct, cb_dct)

    def i_decode(
            self,
            y: npt.NDArray[np.float32],
            cr: npt.NDArray[np.float32],
            cb: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.uint8]:

        # block the frame 
        y_blocks  = y.reshape(y.shape[0] // self.block_size  , self.block_size, -1, self.block_size).swapaxes(1, 2)
        cr_blocks = cr.reshape(cr.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
        cb_blocks = cb.reshape(cb.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)

        # dequantization + 2D IDCT section
        y_idct  = np.zeros(shape= (self.num_row_blocks     , self.num_col_blocks     , self.block_size, self.block_size), dtype= np.float32)
        cr_idct = np.zeros(shape= (self.num_row_blocks // 2, self.num_col_blocks // 2, self.block_size, self.block_size), dtype= np.float32)
        cb_idct = np.zeros(shape= (self.num_row_blocks // 2, self.num_col_blocks // 2, self.block_size, self.block_size), dtype= np.float32)

        # loop over blocks
        for row in range(self.num_row_blocks):
            for col in range(self.num_col_blocks):

                # dequantization + 2D IDCT
                y_idct[row, col]  = fftpack.idctn(y_blocks[row, col]  * (MPEG.q_luminance_table * self.scale), norm= 'ortho')

                if row < self.num_row_blocks // 2 and col < self.num_col_blocks // 2: # chrominance channels are downsampled
                    cr_idct[row, col] = fftpack.idctn(cr_blocks[row, col] * (MPEG.q_chrominance_table * self.scale), norm= 'ortho')
                    cb_idct[row, col] = fftpack.idctn(cb_blocks[row, col] * (MPEG.q_chrominance_table * self.scale), norm= 'ortho')

        # unblock the frame
        y  = y_idct.swapaxes(1, 2).reshape(self.height, self.width)
        cr = cr_idct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)
        cb = cb_idct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)

        # shift intensity [-128, 127] -> [0, 255]
        y  += 128
        cr += 128
        cb += 128

        # upsample chrominance channels
        cr = cv2.resize(cr, (cr.shape[1] * 2, cr.shape[0] * 2))
        cb = cv2.resize(cb, (cb.shape[1] * 2, cb.shape[0] * 2))

        # concatenate channels & clip range to [0, 255]
        reconstructed_ycrcb = cv2.merge((y, cr, cb)).clip(0, 255)

        # convert YCrCb to RGB
        reconstructed_rgb = cv2.cvtColor(reconstructed_ycrcb, cv2.COLOR_YCrCb2BGR)

        # clip range to [0, 255] & change dtype to np.uint8
        reconstructed_rgb = reconstructed_rgb.clip(0, 255).astype(np.uint8)

        return reconstructed_rgb

    def encode(
            self,
            i: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ) -> tuple[npt.NDArray[np.int32], tuple[npt.NDArray[np.float32]], npt.NDArray[np.float32]]:
        
        # decode i-frame & save it
        self.reconstructed_frames[0] = self.i_decode(*i)

        # initialize motion_vectors
        motion_vectors = np.zeros(shape= (self.frames - 1, self.num_row_macroblocks, self.num_col_macroblocks, 2), dtype= np.int32)

        # initialize motion compensate frame (predicting frame)
        p = np.zeros_like(self.actual_frames[0], dtype= np.float32)

        # initialize differences [residuals] for dct coefficients
        diffs_y  = np.zeros(shape= (self.frames - 1, self.height     , self.width)     , dtype= np.float32)
        diffs_cr = np.zeros(shape= (self.frames - 1, self.height // 2, self.width // 2), dtype= np.float32)
        diffs_cb = np.zeros(shape= (self.frames - 1, self.height // 2, self.width // 2), dtype= np.float32)

        # initialize residuals [only on Y channel]
        residuals = np.zeros(shape= (self.frames - 1, self.height, self.width), dtype= np.float32)

        # loop over frames [exclude the i-frame]
        for f in range(1, self.frames):

            # reference & current frame
            reference = self.reconstructed_frames[f-1].astype(np.float32)
            current   = self.actual_frames[f].astype(np.float32)

            # add padding to the reference frame in order to compute motion vectors later
            reference = np.pad(reference, ((self.search_area, self.search_area), (self.search_area, self.search_area), (0, 0)), mode= 'constant', constant_values= 0)

            # RGB to YCrCB
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)
            current   = cv2.cvtColor(current  , cv2.COLOR_BGR2YCrCb)

            # loop over macroblocks to find the best motion vectors
            for row in range(self.num_row_macroblocks):
                for col in range(self.num_col_macroblocks):

                    # reference macroblock extended by search_area
                    reference_mb = reference[
                        row * self.macroblock_size: row * self.macroblock_size + self.macroblock_size + 2 * self.search_area,
                        col * self.macroblock_size: col * self.macroblock_size + self.macroblock_size + 2 * self.search_area,
                        0 # best-matching only computed on Y channel
                    ]

                    # current macroblock
                    current_mb = current[
                        row * self.macroblock_size: row * self.macroblock_size + self.macroblock_size,
                        col * self.macroblock_size: col * self.macroblock_size + self.macroblock_size,
                        0 # best-matching only computed on Y channel
                    ]

                    # mse metric to find the best motion-vectors
                    mse = np.zeros(shape= (self.search_area * 2 + 1, self.search_area * 2 + 1), dtype= np.float32)

                    # loop over pixels of extended reference macroblock
                    for i in range(self.search_area * 2 + 1):
                        for j in range(self.search_area * 2 + 1):
                            roi = reference_mb[i: self.macroblock_size + i, j: self.macroblock_size + j]
                            mse[i, j] = ((roi - current_mb) ** 2).mean()

                    # save the best motion vectors
                    min_mse_index = np.unravel_index(mse.argmin(), mse.shape)
                    motion_vectors[f-1, row, col] = [-(val - self.search_area) for val in min_mse_index]


            # loop over macroblocks to calculate 'predicted frame'
            for row in range(self.num_row_macroblocks):
                for col in range(self.num_col_macroblocks):
                    
                    # find best-match to the current macroblock
                    best_match = reference [
                            row * self.macroblock_size + self.search_area - motion_vectors[f-1, row, col][0]: row * self.macroblock_size + self.macroblock_size + self.search_area - motion_vectors[f-1, row, col][0],
                            col * self.macroblock_size + self.search_area - motion_vectors[f-1, row, col][1]: col * self.macroblock_size + self.macroblock_size + self.search_area - motion_vectors[f-1, row, col][1]
                        ]

                    # update 'predicted frame'
                    p[
                        row * self.macroblock_size: row * self.macroblock_size + self.macroblock_size,
                        col * self.macroblock_size: col * self.macroblock_size + self.macroblock_size,
                    ] = best_match


            # store residuals
            residuals[f-1] = current[:, :, 0] - p[:, :, 0]

            # compute difference [same as residual but applies to all channels]
            diff = current - p

            # split channels
            diff_y, diff_cr, diff_cb = cv2.split(diff)

            # downsample chrominance channels
            diff_cr = cv2.resize(diff_cr, (diff_cr.shape[1] // 2, diff_cr.shape[0] // 2))
            diff_cb = cv2.resize(diff_cb, (diff_cb.shape[1] // 2, diff_cb.shape[0] // 2))

            # blocking the frame
            diff_y  = diff_y.reshape(diff_y.shape[0] // self.block_size  , self.block_size, -1, self.block_size).swapaxes(1, 2)
            diff_cr = diff_cr.reshape(diff_cr.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
            diff_cb = diff_cb.reshape(diff_cb.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)

            # 2D DCT + quantization section
            diff_y_dct  = np.zeros_like(diff_y)
            diff_cr_dct = np.zeros_like(diff_cr)
            diff_cb_dct = np.zeros_like(diff_cb)

            # loop over blocks
            for row in range(self.num_row_blocks):
                for col in range(self.num_col_blocks):

                    # 2D DCT + quantization
                    diff_y_dct[row, col]  = np.round(fftpack.dctn(diff_y[row, col], norm= 'ortho')  / (MPEG.q_luminance_table * self.scale))
                    if row < self.num_row_blocks // 2 and col < self.num_col_blocks // 2:
                        diff_cr_dct[row, col] = np.round(fftpack.dctn(diff_cr[row, col], norm= 'ortho') / (MPEG.q_chrominance_table * self.scale))
                        diff_cb_dct[row, col] = np.round(fftpack.dctn(diff_cb[row, col], norm= 'ortho') / (MPEG.q_chrominance_table * self.scale))

            # unblock the frame
            diff_y  = diff_y_dct.swapaxes(1, 2).reshape(self.height      , self.width)
            diff_cr = diff_cr_dct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)
            diff_cb = diff_cb_dct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)

            # store diffs
            diffs_y[f-1]  = diff_y
            diffs_cr[f-1] = diff_cr
            diffs_cb[f-1] = diff_cb

            # block the frame
            diff_y  = diff_y.reshape(diff_y.shape[0] // self.block_size  , self.block_size, -1, self.block_size).swapaxes(1, 2)
            diff_cr = diff_cr.reshape(diff_cr.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
            diff_cb = diff_cb.reshape(diff_cb.shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)

            # reconstruct diffs to add to the P to reconstruct current frame
            diff_y_idct  = np.zeros_like(diff_y)
            diff_cr_idct = np.zeros_like(diff_cr)
            diff_cb_idct = np.zeros_like(diff_cb)

            # dequantization + 2D IDCT section
            for row in range(self.num_row_blocks):
                for col in range(self.num_col_blocks):

                    # dequantization + 2D IDCT
                    diff_y_idct[row, col] = fftpack.idctn(diff_y[row, col]  * (MPEG.q_luminance_table * self.scale), norm= 'ortho')
                    if row < self.num_row_blocks // 2 and col < self.num_col_blocks // 2:
                        diff_cr_idct[row, col] = fftpack.idctn(diff_cr[row, col] * (MPEG.q_chrominance_table * self.scale), norm= 'ortho')
                        diff_cb_idct[row, col] = fftpack.idctn(diff_cb[row, col] * (MPEG.q_chrominance_table * self.scale), norm= 'ortho')

            # unblock the frame
            diff_y  = diff_y_idct.swapaxes(1, 2).reshape(self.height      , self.width)
            diff_cr = diff_cr_idct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)
            diff_cb = diff_cb_idct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)

            # upsample chrominance channels
            diff_cr = cv2.resize(diff_cr, (diff_cr.shape[1] * 2, diff_cr.shape[0] * 2))
            diff_cb = cv2.resize(diff_cb, (diff_cb.shape[1] * 2, diff_cb.shape[0] * 2))

            # concatenate channels
            reconstructed_diff = cv2.merge((diff_y, diff_cr, diff_cb))

            # convert YCrCb to RGB
            p = cv2.cvtColor(p, cv2.COLOR_YCrCb2BGR)
            reconstructed_diff = cv2.cvtColor(reconstructed_diff, cv2.COLOR_YCrCb2BGR)

            # reconstructed frame [clip range to [0, 255] & change dtype to np.uint8]
            reconstructed_frame = (p + reconstructed_diff).clip(0, 255).astype(np.uint8)

            # store
            self.reconstructed_frames[f] = reconstructed_frame

        return motion_vectors, (diffs_y, diffs_cr, diffs_cb), residuals

    def decode(
            self,
            i: list[npt.NDArray[np.float32], npt.NDArray[np.float32],npt.NDArray[np.float32]],
            mv: npt.NDArray[np.int32],
            diff: list[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.uint8]:

        # store all recostructed frames [np.float32 -> np.uint8 at the end]
        reconstructed_frames = np.zeros_like(self.actual_frames)

        # initialize motion compensate frame (predicting frame)
        p = np.zeros_like(self.actual_frames[0], dtype= np.float32)

        # reconstruct i-frame
        reconstructed_frames[0] = self.i_decode(*i)

        # loop over all p-frames
        for f in range(1, self.frames):

            # reference frame
            reference = reconstructed_frames[f-1].astype(np.float32)

            # add padding to the reference frame in order to compute motion vectors later
            reference = np.pad(reference, ((self.search_area, self.search_area), (self.search_area, self.search_area), (0, 0)), mode= 'constant', constant_values= 0)

            # RGB to YCrCB
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2YCrCb)

            # loop over macroblocks to calculate 'predicted frame'
            for row in range(self.num_row_macroblocks):
                for col in range(self.num_col_macroblocks):

                    # find best-match to the current macroblock
                    best_match = reference [
                            row * self.macroblock_size + self.search_area - mv[f-1, row, col][0]: row * self.macroblock_size + self.macroblock_size + self.search_area - mv[f-1, row, col][0],
                            col * self.macroblock_size + self.search_area - mv[f-1, row, col][1]: col * self.macroblock_size + self.macroblock_size + self.search_area - mv[f-1, row, col][1]
                        ]

                    # update 'predicted frame'
                    p[
                        row * self.macroblock_size: row * self.macroblock_size + self.macroblock_size,
                        col * self.macroblock_size: col * self.macroblock_size + self.macroblock_size,
                    ] = best_match


            # block the differences
            diff_y  = diff[0][f-1].reshape(diff[0][f-1].shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
            diff_cr = diff[1][f-1].reshape(diff[1][f-1].shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)
            diff_cb = diff[2][f-1].reshape(diff[2][f-1].shape[0] // self.block_size, self.block_size, -1, self.block_size).swapaxes(1, 2)

            # reconstruct diffs to add to the P to reconstruct current frame
            diff_y_idct  = np.zeros(shape= (self.num_row_blocks     , self.num_col_blocks     , self.block_size, self.block_size))
            diff_cr_idct = np.zeros(shape= (self.num_row_blocks // 2, self.num_col_blocks // 2, self.block_size, self.block_size))
            diff_cb_idct = np.zeros(shape= (self.num_row_blocks // 2, self.num_col_blocks // 2, self.block_size, self.block_size))

            # dequantization + 2D IDCT section
            for row in range(self.num_row_blocks):
                for col in range(self.num_col_blocks):

                    # dequantization + 2D IDCT
                    diff_y_idct[row, col] = fftpack.idctn(diff_y[row, col]  * (MPEG.q_luminance_table * self.scale), norm= 'ortho')
                    if row < self.num_row_blocks // 2 and col < self.num_col_blocks // 2:
                        diff_cr_idct[row, col] = fftpack.idctn(diff_cr[row, col] * (MPEG.q_chrominance_table * self.scale), norm= 'ortho')
                        diff_cb_idct[row, col] = fftpack.idctn(diff_cb[row, col] * (MPEG.q_chrominance_table * self.scale), norm= 'ortho')


            # unblock the differences
            diff_y  = diff_y_idct.swapaxes(1, 2).reshape(self.height      , self.width)
            diff_cr = diff_cr_idct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)
            diff_cb = diff_cb_idct.swapaxes(1, 2).reshape(self.height // 2, self.width // 2)

            # upsample chrominance channels
            diff_cr = cv2.resize(diff_cr, (diff_cr.shape[1] * 2, diff_cr.shape[0] * 2))
            diff_cb = cv2.resize(diff_cb, (diff_cb.shape[1] * 2, diff_cb.shape[0] * 2))

            # concatenate channels
            reconstructed_diff = cv2.merge((diff_y, diff_cr, diff_cb)).astype(np.float32)

            # convert YCrCb to RGB
            p = cv2.cvtColor(p, cv2.COLOR_YCrCb2BGR)
            reconstructed_diff = cv2.cvtColor(reconstructed_diff, cv2.COLOR_YCrCb2BGR)

            # reconstructed frame [clip range to [0, 255] & change dtype to np.uint8]
            reconstructed_frame = (p + reconstructed_diff).clip(0, 255).astype(np.uint8)

            # store
            reconstructed_frames[f] = reconstructed_frame

        return reconstructed_frames