# 🎛️ Codecs

- A codec is a software or hardware component that **compresses** and **decompresses** digital media files.
- It allows for **efficient storage**, **transmission**, and **playback** of audio and video content.

## 🖼️ **Image Codecs**

### **BPG (Better Portable Graphics)**

A high-efficiency image format designed to provide better compression and quality than JPEG by using the HEVC (High Efficiency Video Coding) standard

- 📥 Download
  - Executable (.exe) files are officially available at the [official site](https://bellard.org/bpg/).
  - Link: [bellard.org/bpg/bpg-0.9.8-win64.zip](https://bellard.org/bpg/bpg-0.9.8-win64.zip)
- 🛠️ Usage
  - To Encode:

    ```bash
    ./bpgenc.exe <input.[jpg|png]> -o <encoded.bpg> -m 9
    ```

  - To Decode:

    ```bash
    ./bpgdec.exe -o <decoded.[ppm|png]> <encoded.bpg>
    ```

- ©️ Source Code
  - Encoder & Decoder: [bellard.org/bpg/libbpg-0.9.8.tar.gz](https://bellard.org/bpg/libbpg-0.9.8.tar.gz).

## 🎥 **Video Codecs**

⚠️ **Notes**

- YUV files are raw, uncompressed video files that store color information in separate channels: Y (luminance), U (chrominance), and V (chrominance).
- Using formats other than .yuv for video files may result in incompatibility and corrupted output videos.
- Use [ffmpeg](https://github.com/BtbN/FFmpeg-Builds) to convert videos to .yuv format:

  ```bash
  ffmpeg -i input.y4m -pix_fmt yuv420p -f rawvideo output.yuv
  ```

### **H.264 (AVC)**

The most widely used video compression standard, offering high quality at low bitrates.

- 📥 Download
  - Executable (.exe) files are compiled directly from the latest source code (jm19.0).
  - Configuration files are copied and edited directly from the latest source code (jm19.0).
  - Link: [drive.google.com/drive/folders/19pNeG4WVZ1w7U_20Dm-4KzNhB_x-5EGv](https://drive.google.com/drive/folders/19pNeG4WVZ1w7U_20Dm-4KzNhB_x-5EGv)
- 🛠️ Usage
  - Create a config file or Edit encoder.cfg based on your goal.
  - To Encode:

    ```bash
    ./lencode.exe -f <encoder.cfg>
    ```

  - To Decode:

    ```bash
    ./ldecode.exe -f <decoder.cfg>
    ```

- ©️ Source Code
  - Encoder & Decoder: [vcgit.hhi.fraunhofer.de/jvet/JM](https://vcgit.hhi.fraunhofer.de/jvet/JM).

### **H.265 (HEVC)**

Successor to H.264, offering even better compression for even higher quality or lower bitrates.

- 📥 Download
  - Executable (.exe) files are compiled directly from the latest source code.
  - Configuration files are copied and edited directly from the latest source code.
  - Link: [drive.google.com/drive/folders/19pNeG4WVZ1w7U_20Dm-4KzNhB_x-5EGv](https://drive.google.com/drive/folders/19pNeG4WVZ1w7U_20Dm-4KzNhB_x-5EGv).
- 🛠️ Usage
  - Create a config file or Edit encoder.cfg based on your goal.
  - To Encode:

    ```bash
    ./TAppEncoder.exe -c <encoder_intra_main.cfg>
    ```

  - To Decode:

    ```bash
    ./TAppDecoder.exe -b <encoded_video.265> -o <decoded_video.yuv>
    ```

- ©️ Source Code
  - Encoder & Decoder: [vcgit.hhi.fraunhofer.de/jvet/HM](https://vcgit.hhi.fraunhofer.de/jvet/HM).

### **H.266 (VVC)**

The latest video compression standard, offering significant efficiency improvements over H.265 for high-resolution streaming and future video applications.

- 📥 Download
  - Executable (.exe) files are compiled directly from the latest source code.
  - Link: [drive.google.com/drive/folders/19pNeG4WVZ1w7U_20Dm-4KzNhB_x-5EGv](https://drive.google.com/drive/folders/19pNeG4WVZ1w7U_20Dm-4KzNhB_x-5EGv).
- 🛠️ Usage
  - To Encode:

    ```bash
    ./vvencapp -i input_video.yuv -o encoded_video.266 -s 176x144 --framerate 30 --frames 20 --profile main_10 --level 5.1 --format yuv420 --bitrate 0 --qp 32
    ```

    ⚠️ Note: for fractional frame rates [e.g. 29.97 fps], replace `--framerate 30` with `--framerate 30000 --framescale 1001` because 29.97 is 30000/1001

  - To Decode:

    ```bash
    ./vvdecapp -b encoded_video.266 -o decoded_video.yuv
    ```

- ©️ Source Code
  - Encoder: [github.com/fraunhoferhhi/vvenc](https://github.com/fraunhoferhhi/vvenc).
  - Decoder: [github.com/fraunhoferhhi/vvdec](https://github.com/fraunhoferhhi/vvdec).
