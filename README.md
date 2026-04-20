# 🌿 Leaf Size Analysis (ArUco + OpenCV)

A Python-based tool for automated leaf measurement using image processing and ArUco markers.

This project is designed for **Linux systems** and processes images to measure leaf dimensions and area based on a calibrated reference frame.

---

# ⚠️ Important: ArUco Marker Setup

This software requires a **fixed reference setup**:

- 4 ArUco markers must be present in every image
- Marker dictionary: `DICT_4X4_50`
- The markers define a physical reference area of:
60 cm (height) × 30 cm (width)


👉 The outer marker corners must match this real-world size.

Without correct marker placement, measurements will fail.

---

# 📊 Features

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

---

# 🐧 Requirements (Linux only)

## System dependency

```bash
git clone https://github.com/Fledermausmann-C3D2/REPO.git
cd REPO
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
