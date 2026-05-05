# Leaf Size Analysis (ArUco + OpenCV)

A Python-based tool for automated leaf measurement using image processing and ArUco markers.

This project is designed for processes images to measure leaf dimensions and area based on a calibrated reference frame.

## Background

At JKI we need to mesuare a lot of leafs by hand and it needs time.
For this i write this program to autmize the work easyly with the right background.

>[!NOTE]
>Its for long leafs like barley or similar 

You can change this at the Morphologie Point

```kernel = np.ones((25,3), np.uint8)```

Maybe change it to:

```kernel = np.ones((10,5), np.uint8)```
for not long leafs...


>[!IMPORTANT]
>ArUco Marker Setup
>This software requires a **fixed reference setup**:
>
>- 4 ArUco markers must be present in every image
>- Marker dictionary: `DICT_4X4_50`
>- The markers define a physical reference area of:
> 52.4 cm (height) × 30 cm (width)
> Change it in :
> `REAL_WIDTH_CM = 30`
> `REAL_HEIGHT_CM = 52.4`
>
>The outer marker corners must match this real-world size.
>
>Without correct marker placement, measurements will fail.


## Features

- Batch processing of image folders
- Automatic ArUco marker detection
- Perspective correction (homography)
- Leaf segmentation using HSV color space
- Measurement of:
  - length (cm)
  - width (cm)
  - area (cm²)
  - angle (deg)
- CSV export of results
- Error logging system
- GUI (ttkbootstrap)


>[!NOTE]
>This Programm is tested on Linux

## System dependency

```bash
git clone https://github.com/Fledermausmann-C3D2/Leaf_sice.git
cd Leaf_Sice
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## New Debugs 

- Debug Windows to control
- Markers will remove for Mesurement
- Reflektions will ignore with HSV Saturation
