# 📷 Media Processing Workshop

[![License](https://img.shields.io/github/license/mr-pylin/media-processing-workshop?color=blue)](https://github.com/mr-pylin/media-processing-workshop/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.13.1-yellow?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3131/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9eb774b7945449cdb86029e9093b3c73)](https://app.codacy.com/gh/mr-pylin/media-processing-workshop/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
![Repo Size](https://img.shields.io/github/repo-size/mr-pylin/media-processing-workshop?color=lightblue)
![Last Updated](https://img.shields.io/github/last-commit/mr-pylin/media-processing-workshop?color=orange)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?color=brightgreen)](https://github.com/mr-pylin/media-processing-workshop/pulls)

A comprehensive resource to explore media processing, from fundamental concepts to advanced techniquess.

## 📖 Table of Contents

### 📘 Foundations

1. [**Complex Numbers & Euler’s Formula**](./code/foundations/01-complex-numbers.ipynb)

### 📖 Main Notebooks

1. [**Introduction to Digital Images**](./code/01-introduction.ipynb)
1. [**Load & Save Images**](./code/02-io.ipynb)
1. [**Interpolating Images**](./code/03-interpolation.ipynb)
1. [**Apply Geometric Transformations**](./code/04-geometric-transformation.ipynb)
1. [**Intensity Transformations**](./code/05-intensity-transformation.ipynb)
1. [**Histogram Processing**](./code/06-histogram.ipynb)
1. [**Spatial Filtering using Convolution**](./code/07-spatial-filtering.ipynb)
1. [**Frequency Filtering using Fourier & Cosine Transform**](./code/08-frequency-filtering.ipynb)
1. [**Multi-Resolution Analysis (Wavelet Transform)**](./code/09-multi-resolution-analysis.ipynb)
1. [**Image Compression (JPEG Coded)**](./code/10-image-compression.ipynb)
1. [**Morphological Processing**](./code/11-morphological-processing.ipynb)

### 📖 Utilities

A collection of concepts and tools utilized in the main notebooks

- [**Color Spaces**](./code/utils/color-space.ipynb)
- [**Introductions to Compression**](./code/utils/compression.ipynb)
- [**Prerequisites and Introductions to Convolution**](./code/utils/convolution.ipynb)
- [**Quality Assessment Metrics**](./code/utils/quality-assessment.ipynb)
- [**Prerequisites and Introductions to Frequency Transforms**](./code/utils/transform.ipynb)

## 📋 Prerequisites

- 👨‍💻 **Programming Fundamentals**
  - Proficiency in **Python** (data types, control structures, functions, classes, etc.).
    - My Python Workshop: [github.com/mr-pylin/python-workshop](https://github.com/mr-pylin/python-workshop)
  - Experience with libraries like **NumPy** and **Matplotlib**.
    - My NumPy Workshop: [github.com/mr-pylin/numpy-workshop](https://github.com/mr-pylin/numpy-workshop)
    - My Data Visualization Workshop: [github.com/mr-pylin/data-visualization-workshop](https://github.com/mr-pylin/data-visualization-workshop)
- 🔣 **Mathematics for Machine Learning**
  - 🔲 **Linear Algebra**: Vectors, matrices, matrix operations.
    - [**Linear Algebra Review and Reference**](https://www.cs.cmu.edu/%7Ezkolter/course/linalg/linalg_notes.pdf) written by [*Zico Kolter*](https://zicokolter.com).
    - [**Notes on Linear Algebra**](https://webspace.maths.qmul.ac.uk/p.j.cameron/notes/linalg.pdf) written by [*Peter J. Cameron*](https://cameroncounts.github.io/web).
    - [**MATH 233 - Linear Algebra I Lecture Notes**](https://www.geneseo.edu/~aguilar/public/assets/courses/233/main_notes.pdf) written by [*Cesar O. Aguilar*](https://www.geneseo.edu/~aguilar/).
  - 🎲 **Probability & Statistics**: Probability distributions, mean/variance, etc.
    - [**MATH1024: Introduction to Probability and Statistics**](https://www.sujitsahu.com/teach/2020_math1024.pdf) written by [*Sujit Sahu*](https://www.southampton.ac.uk/people/5wynjr/professor-sujit-sahu).
- 📶 **Digital Signal Processing Knowledge**
  - [**Digital Image Processing (4th Edition)**](https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm) written by [*Rafael C. Gonzalez*](https://www.imageprocessingplace.com/root_files_V3/about_the_authors/gonzalez.htm) & [*Richard E. Woods*](https://www.imageprocessingplace.com/root_files_V3/about_the_authors/woods.htm)
  - [**The Scientist and Engineer's Guide to Digital Signal Processing**](https://www.dspguide.com/pdfbook.htm) written by [*Steven W. Smith*](https://www.dspguide.com/swsmith.htm)

## ⚙️ Setup

This project requires Python **v3.10** or higher. It was developed and tested using Python **v3.13.1**. If you encounter issues running the specified version of dependencies, consider using this version of Python.

### 📝 List of Dependencies

[![imagecodecs](https://img.shields.io/badge/imagecodecs-2025.3.30-008080)](https://pypi.org/project/imagecodecs/2025.3.30/)
[![ipykernel](https://img.shields.io/badge/ipykernel-6.29.5-ff69b4)](https://pypi.org/project/ipykernel/6.29.5/)
[![ipywidgets](https://img.shields.io/badge/ipywidgets-8.1.5-ff6347)](https://pypi.org/project/ipywidgets/8.1.5/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.10.1-green)](https://pypi.org/project/matplotlib/3.10.1/)
[![numpy](https://img.shields.io/badge/numpy-2.2.4-orange)](https://pypi.org/project/numpy/2.2.4/)
[![opencv-contrib-python](https://img.shields.io/badge/opencv--contrib--python-4.11.0.86-blue)](https://pypi.org/project/opencv-contrib-python/4.11.0.86/)
[![pillow](https://img.shields.io/badge/pillow-11.1.0-cyan)](https://pypi.org/project/Pillow/11.1.0/)
[![scikit-image](https://img.shields.io/badge/scikit--image-0.25.2-darkblue)](https://pypi.org/project/scikit-image/0.25.2/)
[![scipy](https://img.shields.io/badge/scipy-1.15.2-purple)](https://pypi.org/project/scipy/1.15.2/)

### 📦 Installing Dependencies

#### 📦 Method 1: Poetry (**Recommended** ✅)

Use [**Poetry**](https://python-poetry.org/) for dependency management. It handles dependencies, virtual environments, and locking versions more efficiently than pip.  
To install exact dependency versions specified in [**poetry.lock**](./poetry.lock) for consistent environments **without** installing the current project as a package:

```bash
poetry install --no-root
```

#### 📦 Method 2: Pip

Install all dependencies listed in [**requirements.txt**](./requirements.txt) using [**pip**](https://pip.pypa.io/en/stable/installation/):

```bash
pip install -r requirements.txt
```

### 🛠️ Usage Instructions

1. Open the root folder with [**VS Code**](https://code.visualstudio.com/) (`Ctrl/Cmd + K` followed by `Ctrl/Cmd + O`).
1. Open `.ipynb` files using the [**Jupyter extension**](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) integrated with **VS Code**.
1. Select the correct Python kernel and virtual environment where the dependencies were installed.
1. Allow **VS Code** to install any recommended dependencies for working with Jupyter Notebooks.

✍️ **Notes**:  

- It is **highly recommended** to stick with the exact dependency versions specified in [**poetry.lock**](./poetry.lock) or [**requirements.txt**](./requirements.txt) rather than using the latest package versions. The repository has been **tested** on these versions to ensure **compatibility** and **stability**.
- This repository is **actively maintained**, and dependencies are **updated regularly** to the latest **stable** versions.
- The **table of contents** embedded in the **notebooks** may not function correctly on **GitHub**.
- For an improved experience, open the notebooks **locally** or view them via [**nbviewer**](https://nbviewer.org/github/mr-pylin/media-processing-workshop).

## 🔗 Usefull Links

### Tools

- **ffmpeg & ffprobe**:
  - ffmpeg is a Swiss Army knife for media, converting and manipulating audio and video files in a wide range of formats.
  - Link: [github.com/BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds)

- **Video Quality Measurement Tool (VQMT)**:
  - It is a software program designed to analyze the quality of digital video and images.
  - Link: [compression.ru/video/quality_measure](http://www.compression.ru/video/quality_measure/vqmt_download.html)

- **yuv-player**:
  - Lightweight YUV player which supports various YUV format.
  - Link: [github.com/Tee0125/yuvplayer](https://github.com/Tee0125/yuvplayer)

### Benchmark Files

- **DIP3/e — Book Images**
  - A collection of all images and videos used in the **Digital Image Processing (3rd Edition)** book written by [*Rafael C. Gonzalez*](https://www.imageprocessingplace.com/root_files_V3/about_the_authors/gonzalez.htm) & [*Richard E. Woods*](https://www.imageprocessingplace.com/root_files_V3/about_the_authors/woods.htm).
  - Permission is required from the owner of a © image if the image is used for other than personal educational or research purposes.
  - Link: [imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm](https://www.imageprocessingplace.com/DIP-3E/dip3e_book_images_downloads.htm)

- **YUV4MPEG Videos**:
  - Derf's video collection provides uncompressed YUV4MPEG clips for testing video codecs.
  - Link: [media.xiph.org/video/derf](https://media.xiph.org/video/derf/)

### Codecs

- Codecs are algorithms used to **compress** and **decompress** signals, ensuring **efficient storage and transmission of high-quality** signals **e.g. videos**.
- For detailed information on popular image/video codecs, refer to the [**./codecs/README.md**](./codecs/README.md).

### **NumPy**

- A fundamental package for scientific computing in Python, providing support for **arrays**, **matrices**, and a large collection of **mathematical functions**.
- Official site: [numpy.org](https://numpy.org/)

### **Data Visualization**

- A comprehensive collection of Python libraries for creating static, animated, and interactive visualizations: **Matplotlib**, **Seaborn**, and **Plotly**.
- Official sites: [matplotlib.org](https://matplotlib.org/) | [seaborn.pydata.org](https://seaborn.pydata.org/) | [plotly.com](https://plotly.com/)

### **OpenCV (Open Source Computer Vision Library)**

- A powerful open-source library (primarily written in C++) for computer vision and image processing tasks.
- Supports a wide range of functionalities, including image and video processing, object detection, facial recognition, and more.
- Compatible with multiple programming languages, including Python, C++, and Java.
- Official sites: [opencv.org](https://opencv.org/)

## 🔍 Find Me

Any mistakes, suggestions, or contributions? Feel free to reach out to me at:

- 📍[**linktr.ee/mr_pylin**](https://linktr.ee/mr_pylin)

I look forward to connecting with you! 🏃‍♂️

## 📄 License

This project is licensed under the **[Apache License 2.0](./LICENSE)**.  
You are free to **use**, **modify**, and **distribute** this code, but you **must** include copies of both the [**LICENSE**](./LICENSE) and [**NOTICE**](./NOTICE) files in any distribution of your work.

### ©️ Copyright Information

- **Original Images**:
  - The images located in the [./assets/images/original/](./assets/images/original/) folder are licensed under the **[CC BY-ND 4.0](./assets/images/original/LICENSE)**.
  - Note: This license restricts derivative works, meaning you may share these images but cannot modify them.

- The images located in the [./assets/images/dip_3rd/](./assets/images/dip_3rd/) folder are licensed under the table below:  

  | Image                                                                                      | Copyright Owner                            | Address                               |
  |--------------------------------------------------------------------------------------------|--------------------------------------------|---------------------------------------|
  | [CH02_Fig0222(b)(cameraman).tif](./assets/images/dip_3rd/CH02_Fig0222(b)(cameraman).tif)    | Massachusetts Institute of Technology      | [MIT.edu](https://MIT.edu)           |
  | [CH03_Fig0309(a)(washed_out_aerial_image).tif](./assets/images/dip_3rd/CH03_Fig0309(a)(washed_out_aerial_image).tif) | NASA                                       | [nasa.gov](https://nasa.gov)         |
  | [CH03_Fig0326(a)(embedded_square_noisy_512).tif](./assets/images/dip_3rd/CH03_Fig0326(a)(embedded_square_noisy_512).tif) | -                                          | [imageprocessingplace.com](https://imageprocessingplace.com) |
  | [CH03_Fig0354(a)(einstein_orig).tif](./assets/images/dip_3rd/CH03_Fig0354(a)(einstein_orig).tif) | Public domain                              | -                                     |
  | [CH06_Fig0638(a)(lenna_RGB).tif](./assets/images/dip_3rd/CH06_Fig0638(a)(lenna_RGB).tif)    | Public domain                              | -                                     |
  | [CH06_FigP0606(color_bars).tif](./assets/images/dip_3rd/CH06_FigP0606(color_bars).tif)      | -                                          | -                                     |

- **Third-Party Assets**:

  - Additional images located in [./assets/images/third_party/](./assets/images/third_party/) are used with permission or according to their original licenses.
  - Attributions and references to the files included in [./assets/images/third_party/](./assets/images/third_party/) are included in the code where these images are used.

- **Miscellaneous assets**:

  - The images found in [./assets/images/misc/](./assets/images/misc/) are modified versions of the ones listed above.
