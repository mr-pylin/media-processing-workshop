# 📷 Media Processing Workshop
Implementation of various concepts around Digital Media (Image/Video) Processing (DMP) topics.

## 📖 Table of Contents
### Main Notebooks
   1. [**Introduction**](./codes/01_introduction.ipynb)  
   How to read/plot an image using [matplotlib](https://matplotlib.org/) package   
   Image properties: color, dtype, depth, resolution, ...
   1. [**Basic Modifications**](./codes/02_basic-modification.ipynb)  
   Crop, Flip, Circular Shift, Rotation
   1. [**Interpolations**](./codes/03_interpolations.ipynb)  
   Nearest Neighbor, BiLinear, BiCubic, Lanczos interpolation
   1. [**Intensity Transformation**](./codes/04_intensity-transformation.ipynb)  
   Negative, Logarithm, Power-Law (Gamma correction), Piecewise-Linear Transform
   1. [**Histogram**](./codes/05_histogram.ipynb)  
   Histogram Stretching, Shrinking, Sliding  
   Global Histogram Equalization  
   Local Histogram Equalization (Adaptive Histogram Equalization)  
   Adaptive Contrast Enhancement (ACE)  
   Histogram Matching (Specification)  
   1. [**Convolution**](./codes/06_convolution.ipynb)  
   1D Convolution  
   2D Convolution (GrayScale/RGB image)  
   1. [**Fourier Transform**](./codes/07_fourier-transform.ipynb)  
   Basis vectors(1D)/images(2D)  
   Forward/Backward Fourier Transform    
   Fast Fourier Transform (FFT)  
   Ideal Low-Pass filter  
   Cardinal Sine (sinc) filter  
   Ringing Effect  
   Shift, Rotation, Flip effect in frequency domain  
   Image sharpening using a gaussian high-pass filter  
   Periodic noise removal  
   1. [**Cosine Transform**](./codes/08_cosine-transform.ipynb)  
   Basis vectors(1D)/images(2D)  
   Forward/Backward Cosine Transform  
   Compression Effect (DFT vs DCT)  
   Zonal Masking  
   1. [**Quality Assessment**](./codes/09_quality-assessment.ipynb)  
   Mean Squared Error (MSE)  
   Signal-to-Noise Ratio (SNR)  
   Peak Signal-to-Noise Ratio (PSNR)  
   Structural Similarity Index (SSIM)  
   Root Mean Square Error (RMSE)  
   Mean Absolute Error (MAE)  
   Mean Structural Similarity Index (MSSIM)  
   Visual Information Fidelity (VIF)  
   Feature Similarity Index (FSIM)  
   Multi-Scale Structural Similarity Index (MS-SSIM)  
   1. [**Steganography**](./codes/10_least-significant-bit-steganography.ipynb)  
   Steganography using least significant bits
   1. [**JPEG codec**](./codes/11_jpeg-codec.ipynb)  
   JPEG Encoder & Decoder
   1. [**MPEG codec**](./codes/12_mpeg-codec.ipynb)  
   MPEG Encoder & Decoder
   1. [**Image Registration**](./codes/13_image-registration.ipynb)  
   Aligning multiple images into a common coordinate system
   1. [**Image Stitching**](./codes/14_image-stitching.ipynb)  
   Combining multiple images to create a single larger image [Panorama]
   1. [**Optical Flow**](./codes/15_optical-flow.ipynb)  
   Optical Flow using Lucas-Kanade & Farneback algorithms

### Utilities
Implementation of several concepts utilized in the main notebooks
   - [**Padding and Convolution**](./codes/utils/convolution.py)  
   Provides utility functions for 1D and 2D convolution, along with flexible padding options.
   - [**DCT Implementation**](./codes/utils/dct.py)  
   Discrete Cosine Transform (DCT) implementation for 1D and 2D signals.
   - [**DFT Implementation**](./codes/utils/dft.py)  
   Implementations of 1D and 2D Discrete Fourier Transform (DFT) and related functions.
   - [**Filter Functions Implementation**](./codes/utils/filters.py)  
   Implementations of various 2D filter functions including ideal, Gaussian, sinc, Butterworth, Chebyshev, Bessel, and block masks.
   - [**JPEG Codec Implementation**](./codes/utils/jpeg.py)  
   Class-based implementation of JPEG encoder and decoder using discrete cosine transform (DCT) and quantization.
   - [**MPEG Codec Implementation**](./codes/utils/mpeg.py)  
   Class-based implementation of MPEG encoder and decoder using discrete cosine transform (DCT) and quantization.
   - [**Quality Assessment Metrics**](./codes/utils/quality_assessment.py)  
   Implementation of famous metrics e.g. MSE, SNR, PSNR, SSIM, RMSE, MAE, ... 
   - [**Spatial Modifications**](./codes/utils/spatial_modification.py)  
   Implementation of concepts in spatial domain e.g. interpolations and histograms
   - [**Steganography**](./codes/utils/steganography.py)  
   Implementation of a simple steganography method using Least Significant Bits (LSB)

## 📋 Prerequisites
   - **Programming Fundamentals**
      - Proficiency in **Python** (data types, control structures, functions, etc.).
         - My Python Workshop: [github.com/mr-pylin/python-workshop](https://github.com/mr-pylin/python-workshop)
      - Experience with libraries like **NumPy**, **Matplotlib** and **OpenCV**.
         - My NumPy Workshop: [github.com/mr-pylin/numpy-workshop](https://github.com/mr-pylin/numpy-workshop)
         - My MatPlotLib Workshop: [Coming Soon](https://github.com/mr-pylin/#)
         - My OpenCV Workshop: [Coming Soon](https://github.com/mr-pylin/#)
   - **Digital Signal Processing Knowledge**
      - [*Digital Image Processing (4th Edition)*](https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm) written by [Rafael C. Gonzalez](https://www.imageprocessingplace.com/root_files_V3/about_the_authors/gonzalez.htm) & [Richard E. Woods](https://www.imageprocessingplace.com/root_files_V3/about_the_authors/woods.htm)
      - [*The Scientist and Engineer's Guide to Digital Signal Processing*](https://www.dspguide.com/pdfbook.htm) written by [Steven W. Smith](https://www.dspguide.com/swsmith.htm)
   - **Mathematics for Image Processing**
      - Linear Algebra: Understanding of vectors, matrices, and matrix operations, crucial for transformations, convolutions, and Fourier analysis.
         - [*Linear Algebra Review and Reference*](https://www.cs.cmu.edu/%7Ezkolter/course/linalg/linalg_notes.pdf) written by [Zico Kolter](https://zicokolter.com)
         - [*Notes on Linear Algebra*](https://webspace.maths.qmul.ac.uk/p.j.cameron/notes/linalg.pdf) written by [Peter J. Cameron](https://cameroncounts.github.io/web)
         - [*MATH 233 - Linear Algebra I Lecture Notes*](https://www.geneseo.edu/~aguilar/public/assets/courses/233/main_notes.pdf) written by [Cesar O. Aguilar](https://www.geneseo.edu/~aguilar/)
      - Probability & Statistics: Probability distributions, mean/variance, etc.
         - [*MATH1024: Introduction to Probability and Statistics*](https://www.sujitsahu.com/teach/2020_math1024.pdf) written by [Sujit Sahu](https://www.southampton.ac.uk/people/5wynjr/professor-sujit-sahu)

# 📝 TODO
   - [ ] 04: Adaptive Contrast Enhancement (ACE)
   - [ ] 04: Histogram Matching (Specification)
   - [ ] 14: Sparse Optical Flow using Lucas-Kanade

# ⚙️ Setup
This project was developed using Python `v3.12.3`. If you encounter issues running the specified version of dependencies, consider using this specific Python version.

## 📦 Installing Dependencies
You can install all dependencies listed in `requirements.txt` using [pip](https://pip.pypa.io/en/stable/installation/).
```bash
pip install -r requirements.txt
```

## 🛠️ Usage Instructions
   - Open the root folder with [VS Code](https://code.visualstudio.com/)
      - **Windows/Linux**: `Ctrl + K` followed by `Ctrl + O`
      - **macOS**: `Cmd + K` followed by `Cmd + O`
   - Open `.ipynb` files using [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) integrated with **VS Code**
   - Allow **VS Code** to install any recommended dependencies for working with Jupyter Notebooks.
   - Note: Jupyter is integrated with both **VS Code** & **[Google Colab](https://colab.research.google.com/)**

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
   - **NumPy**
      - A fundamental package for scientific computing in Python, providing support for arrays, matrices, and a large collection of mathematical functions.
      - Official site: [numpy.org](https://numpy.org)
   - **MatPlotLib**:
      - A comprehensive library for creating static, animated, and interactive visualizations in Python
      - Official site: [matplotlib.org](https://matplotlib.org)
   - **OpenCV**:
      - A powerful library for computer vision and image processing, supporting real-time operations on images and videos in Python and other languages.
      - Official site: [opencv.org](https://opencv.org)

# 🔍 Find Me
Any mistakes, suggestions, or contributions? Feel free to reach out to me at:
   - 📍[linktr.ee/mr_pylin](https://linktr.ee/mr_pylin)
   
I look forward to connecting with you! 🏃‍♂️

# ©️ Copyright Information
- **Digital Image Processing by Gonzalez & Woods**:
   - The images located in the [./assets/images/dip_3rd/](./assets/images/dip_3rd/) folder are licensed under the table below.
   - Resources are available for `personal educational or research purposes` at [imageprocessingplace.com](https://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm).
<table style="margin: 0 auto;">
   <thead>
      <tr>
         <th>Image</th>
         <th>Copyright Owner</th>
         <th>Address</th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><a href="./assets/images/dip_3rd/CH02_Fig0222(b)(cameraman).tif">CH02_Fig0222(b)(cameraman).tif</a></td>
         <td>Massachusetts Institute of Technology</td>
         <td><a href="https://MIT.edu">MIT.edu</a></td>
      </tr>
      <tr>
         <td><a href="./assets/images/dip_3rd/CH03_Fig0309(a)(washed_out_aerial_image).tif">CH03_Fig0309(a)(washed_out_aerial_image).tif</a></td>
         <td>NASA</td>
         <td><a href="https://nasa.gov">nasa.gov</a></td>
      </tr>
      <tr>
         <td><a href="./assets/images/dip_3rd/CH03_Fig0326(a)(embedded_square_noisy_512).tif">CH03_Fig0326(a)(embedded_square_noisy_512).tif</a></td>
         <td>-</td>
         <td><a href="https://imageprocessingplace.com">imageprocessingplace.com</a></td>
      </tr>
      <tr>
         <td><a href="./assets/images/dip_3rd/CH03_Fig0354(a)(einstein_orig).tif">CH03_Fig0354(a)(einstein_orig).tif</a></td>
         <td>Public domain</td>
         <td>-</td>
      </tr>
      <tr>
         <td><a href="./assets/images/dip_3rd/CH06_Fig0638(a)(lenna_RGB).tif">CH06_Fig0638(a)(lenna_RGB).tif</a></td>
         <td>Public domain</td>
         <td>-</td>
      </tr>
      <tr>
         <td><a href="./assets/images/dip_3rd/CH06_FigP0606(color_bars).tif">CH06_FigP0606(color_bars).tif</a></td>
         <td>-</td>
         <td>-</td>
      </tr>
   </tbody>
</table>

- **Third-Party Assets**:
   - Additional images located in [./assets/images/third_party/](./assets/images/third_party/) are used with permission or according to their original licenses.
   - Attributions and references to original sources are included in the code where these images are used.
<table style="margin: 0 auto;">
   <tr>
      <th>Image</th>
      <th>Copyright Owner</th>
      <th>Address</th>
   </tr>
   <tr>
      <td><a href="./assets/images/third_party/nature_1.jpg">nature_1.jpg</a></td>
      <td>-</td>
      <td><a href="https://www.pexels.com/photo/areal-view-of-lake-bridge-and-trees-during-daytime-145525/">pexels.com</a></td>
   </tr>
   <tr>
      <td><a href="./assets/images/third_party/nature_2.jpg">nature_2.jpg</a></td>
      <td>-</td>
      <td><a href="https://www.pexels.com/photo/areal-view-of-lake-bridge-and-trees-during-daytime-145525/">pexels.com</a></td>
   </tr>
</table>

- **Miscellaneous assets**:
<table style="margin: 0 auto;">
   <tr>
      <th>Image</th>
      <th>Copyright Owner</th>
      <th>Address</th>
   </tr>
   <tr>
      <td><a href="./assets/images/misc/keyboard_1.jpg">keyboard_1.jpg</a></td>
      <td>Amirhossein Heydari</td>
      <td><a href="https://github.com/mr-pylin">github.com/mr-pylin</a></td>
   </tr>
   <tr>
      <td><a href="./assets/images/misc/keyboard_2.jpg">keyboard_2.jpg</a></td>
      <td>Amirhossein Heydari</td>
      <td><a href="https://github.com/mr-pylin">github.com/mr-pylin</a></td>
   </tr>
   <tr>
      <td><a href="./assets/images/misc/test.tif">test.tif</a></td>
      <td>Amirhossein Heydari</td>
      <td><a href="https://github.com/mr-pylin">github.com/mr-pylin</a></td>
   </tr>
</table>

# 📄 License
This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to use, modify, and distribute this code, but you must include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.  
Note: Assets in the above tables may have their own licenses