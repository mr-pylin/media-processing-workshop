# 📷 Digital Media Processing
Implementation of various concepts around Digital Media (Image/Video) Processing (DMP) topics.

## 📋 Table of Contents
0.  [Introduction](00_introduction.ipynb)
      - How to read/load an image using [matplotlib](https://matplotlib.org/)
      - How to plot an image using [matplotlib](https://matplotlib.org/)
      - Image properties: color, dtype, depth, resolution, ...
1.  [Basic Modifications](01_basic-modification.ipynb)
      - Crop, Flip, Circular Shift, Rotation
2.  [Interpolations](02_interpolations.ipynb)
      - Nearest Neighbor interpolation
      - BiLinear interpolation
      - BiCubic interpolation
      - Lanczos interpolation
3.  [Intensity Transformation](03_intensity-transformation.ipynb)
      - Negative Transform
      - Logarithm Transform
      - Power-Law (Gamma correction) Transform
      - Piecewise-Linear
4.  [Histogram](04_histogram.ipynb)
      - Histogram Stretching, Shrinking, Sliding
      - Global Histogram Equalization
      - Local Histogram Equalization (Adaptive Histogram Equalization)
      - Adaptive Contrast Enhancement (ACE)
      - Historam Matching (Specification)
5.  [Convolution](05_convolution.ipynb)
      - 1D Convolution
      - 2D Convolution
         - Convolution on GrayScale Image
         - Convolution on RGB Image
6.  [Fourier Transform](06_fourier-transform.ipynb)
      - Basis vectors(1D) / images(2D)
      - Forward Fourier Transform
      - Backward Fourier Transform
      - Fast Fourier Transform
      - Ideal Low-Pass filter
      - cardinal sine (sinc) filter
      - Ringing Effect
      - Shift, Rotation, Flip effect in frequency domain
      - Image sharpening using a gaussian high-pass filter
      - Periodic noise removal
7.  [Cosine Transform](07_cosine-transform.ipynb)
      - Basis vectors(1D) / images(2D)
      - Forward Cosine Transform
      - Backward Cosine Transform
      - Compression Effect (DFT vs DCT)
      - Zonal Masking
8.  [Quality Assessment](08_quality-assessment.ipynb)
      - Mean Squared Error (MSE)
      - Signal-to-Noise Ratio (SNR)
      - Peak Signal-to-Noise Ratio (PSNR)
      - Structural Similarity Index (SSIM)
      - Root Mean Square Error (RMSE)
      - Mean Absolute Error (MAE)
      - Mean Structural Similarity Index (MSSIM)
      - Visual Information Fidelity (VIF)
      - Feature Similarity Index (FSIM)
      - Multi-Scale Structural Similarity Index (MS-SSIM)
9.  [Steganography](09_least-significant-bit-steganography.ipynb)
      - Steganography using least significant bits
10. [JPEG codec](10_jpeg-codec.ipynb)
      - JPEG Encoder
      - JPEG Decoder
11. [MPEG codec](11_mpeg-codec.ipynb)
      - MPEG Encoder
      - MPEG Decoder
12. [Image Registration](12_image-registration.ipynb)
      - Aligning multiple images into a common coordinate system
13. [Image Stitching](13_image-stitching.ipynb)
      - Combining multiple images to create a single larger image [Panorama]
14. [Optical Flow](14_optical-flow.ipynb)
      - Lucas-Kanade
      - Farneback

## 📦 Installing Dependencies
You can install all dependencies listed in `requirements.txt` using [pip](https://pip.pypa.io/en/stable/installation/).
```bash
pip install -r requirements.txt
```
**Note:** This project was developed using Python `v3.12.3`. If you encounter issues running the dependencies or code, consider using this specific Python version.

## 🛠️ Usage
   - Open the root folder with [VS Code](https://code.visualstudio.com/)
      - **Windows/Linux**: `Ctrl + K` followed by `Ctrl + O`
      - **macOS**: `Cmd + K` followed by `Cmd + O`
   - Open `.ipynb` files using [Jupyter](https://jupyter.org/) extension integrated with VS Code
   - Allow Visual Studio Code to install any recommended dependencies for working with Jupyter Notebooks.
   - Jupyter is integrated with both [VS Code](https://code.visualstudio.com/) & [Google Colab](https://colab.research.google.com/)

## 🔍 Find Me
Any mistakes, suggestions, or contributions? Feel free to reach out to me at:
   - 📍[linktr.ee/mr_pylin](https://linktr.ee/mr_pylin)
   
I look forward to connecting with you! 

## 🔗 Usefull Links
   - **ffmpeg & ffprobe**:
      - ffmpeg is a Swiss Army knife for media, converting and manipulating audio and video files in a wide range of formats.
      - Link: [github.com/BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)
   - **YUV4MPEG Videos**:
      - Derf's video collection provides uncompressed YUV4MPEG clips for testing video codecs.
      - Link: [media.xiph.org/video/derf](https://media.xiph.org/video/derf/)
   - **Video Quality Measurement Tool (VQMT)**:
      - It is a software program designed to analyze the quality of digital video and images.
      - Link: [compression.ru/video/quality_measure](http://www.compression.ru/video/quality_measure/vqmt_download.html)
   - **yuv-player**:
      - Lightweight YUV player which supports various YUV format.
      - Link: [github.com/Tee0125/yuvplayer](https://github.com/Tee0125/yuvplayer)
   - **H.264 (AVC) codec**:
      - The most widely used video compression standard, offering high quality at low bitrates.
      - Link: [vcgit.hhi.fraunhofer.de/jvet/JM](https://vcgit.hhi.fraunhofer.de/jvet/JM)
   - **H.265 (HEVC) codec**:
      - Successor to H.264, offering even better compression for even higher quality or lower bitrates.
      - Link: [vcgit.hhi.fraunhofer.de/jvet/HM](https://vcgit.hhi.fraunhofer.de/jvet/HM)
   - **H.266 (VVC) codec**:
      - The latest video compression standard, offering significant efficiency improvements over H.265 for high-resolution streaming and future video applications.
      - Encoder link: [github.com/fraunhoferhhi/vvenc](https://github.com/fraunhoferhhi/vvenc)
      - Decoder link: [github.com/fraunhoferhhi/vvdec](https://github.com/fraunhoferhhi/vvdec)

## 📝 TODO
   - [ ] 04: Adaptive Contrast Enhancement (ACE)
   - [ ] 04: Historam Matching (Specification)
   - [ ] 14: Sparse Optical Flow using Lucas-Kanade

## ©️ Resource Credits
   - Most of the images are taken from the book [Digital Image Processing](https://www.amazon.com/Digital-Image-Processing-3Rd-Edn/dp/9332570329), 3rd Edition, by `Rafael C. Gonzalez` and `Richard E. Woods`.
   - Resources are available for `personal educational or research purposes` at [imageprocessingplace.com](https://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm)

| Image                                                                                                    | Copyright Owner                       | Address                                                                                               |
|----------------------------------------------------------------------------------------------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------|
| [CH02_Fig0222(b)(cameraman)](./assets/images/CH02_Fig0222(b)(cameraman).tif)                                 | Massachusetts Institute of Technology | [MIT.edu](https://MIT.edu)                                                                            |
| [CH03_Fig0309(a)(washed_out_aerial_image)](./assets/images/CH03_Fig0309(a)(washed_out_aerial_image).tif)     | NASA                                  | [nasa.gov](https://nasa.gov)                                                                          |
| [CH03_Fig0326(a)(embedded_square_noisy_512)](./assets/images/CH03_Fig0326(a)(embedded_square_noisy_512).tif) | -                                     | [imageprocessingplace.com](https://imageprocessingplace.com)                                          |
| [CH03_Fig0354(a)(einstein_orig)](./assets/images/CH03_Fig0354(a)(einstein_orig).tif)                         | Public domain                         | -                                                                                                     |
| [CH06_Fig0638(a)(lenna_RGB)](./assets/images/CH06_Fig0638(a)(lenna_RGB).tif)                                 | Public domain                         | -                                                                                                     |
| [CH06_FigP0606(color_bars)](./assets/images/CH06_FigP0606(color_bars).tif)                                   | -                                     | -                                                                                                     |
| [horse](./assets/images/horse.gif)                                                                           | -                                     | -                                                                                                     |
| [keyboard_1](./assets/images/keyboard_1.jpg)                                                                 | Amirhossein Heydari                   | [github.com/mr-pylin](https://github.com/mr-pylin)                                                    |
| [keyboard_2](./assets/images/keyboard_2.jpg)                                                                 | Amirhossein Heydari                   | [github.com/mr-pylin](https://github.com/mr-pylin)                                                    |
| [nature_1](./assets/images/nature_1.jpg)                                                                     | -                                     | [pexels.com](https://www.pexels.com/photo/areal-view-of-lake-bridge-and-trees-during-daytime-145525/) |
| [nature_2](./assets/images/nature_2.jpg)                                                                     | -                                     | [pexels.com](https://www.pexels.com/photo/areal-view-of-lake-bridge-and-trees-during-daytime-145525/) |
| [test](./assets/images/test.tif)                                                                             | Amirhossein Heydari                   | [github.com/mr-pylin](https://github.com/mr-pylin)                                                    |