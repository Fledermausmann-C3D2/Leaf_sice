import os
import cv2
import numpy as np
import csv
import re
import threading
import ttkbootstrap as ttk
from tkinter import filedialog


#============== Sorting images =====================
def natural_key(text):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.findall(r'\d+|\D+', text)]

#================== Error logging =============
log_messages = []
error_count = 0

# ================= Real size of 4 markers =================
REAL_WIDTH_CM = 52.4
REAL_HEIGHT_CM = 30

# ================= GUI FUNCTIONS =================

def choose_folder():
    folder = filedialog.askdirectory()
    folder_var.set(folder)

def choose_output():
    file = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV file","*.csv")]
    )
    output_var.set(file)

def start_analysis():
    thread = threading.Thread(target=run_analysis)
    thread.start()

# ================= CORE =================

def order_points(pts):
    pts = np.array(pts)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def run_analysis():

    global error_count, log_messages
    error_count = 0
    log_messages = []

    image_folder = folder_var.get()

    if not image_folder:
        status_var.set("please choose picture folder")
        return

    output_csv = output_var.get()
    if not output_csv:
        output_csv = os.path.join(image_folder, "leaf_measurements.csv")

    # ArUco setup
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
   
    parameters = aruco.DetectorParameters()
    # optional: finer marker detection -->

    #parameters.adaptiveThreshWinSizeMin = 3
    #parameters.adaptiveThreshWinSizeMax = 23
    #parameters.adaptiveThreshWinSizeStep = 10
    #parameters.adaptiveThreshConstant = 7 #(noise)

    image_paths = []

    for root_dir, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".jpg",".png",".jpeg")):
                image_paths.append(os.path.join(root_dir,file))

    image_paths.sort(key=lambda p: natural_key(os.path.basename(p)))

    total_images = len(image_paths)
    progress["maximum"] = total_images

    results = []

    for i, path in enumerate(image_paths):

        file = os.path.basename(path)
        img = cv2.imread(path)
        original = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        # ================= FIX: Markes remove in ORIGINAL  =================
        if ids is not None:
            for c in corners:
                pts = np.int32(c[0])
                cv2.fillConvexPoly(img, pts, (255,255,255))  # weiß übermalen

        if ids is None or len(corners) < 4:
            msg = f"{file}: ❌ Marker missing"

            print(msg)
            log_messages.append(msg)
            error_count += 1

            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()

            continue

        # collect all points
        pts_all = []
        for c in corners:
            for p in c[0]:
                pts_all.append(p)

        pts_all = np.array(pts_all)

        # determine outer shape
        hull = cv2.convexHull(pts_all)

        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) != 4:
            msg = f"{file}: ❌ no 4 corners"

            print(msg)
            log_messages.append(msg)
            error_count += 1

            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()

            continue

        rect = order_points(approx.reshape(4,2))

        # target coordinate system
        scale = 20 # scaling factor for area calculation 10 - 30
        dst = np.array([
            [0,0],
            [REAL_WIDTH_CM*scale,0],
            [REAL_WIDTH_CM*scale,REAL_HEIGHT_CM*scale],
            [0,REAL_HEIGHT_CM*scale]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (int(REAL_WIDTH_CM*scale), int(REAL_HEIGHT_CM*scale)))

      # ================= LEAF =================
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_green = np.array([10, 20, 15])
        upper_green = np.array([100, 255, 255])

        mask_color = cv2.inRange(hsv, lower_green, upper_green)
        mask_sat = (hsv[:,:,1] > 30).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask_color, mask_sat)

        kernel = np.ones((7,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, np.ones((1,1), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_contours = [c for c in contours if cv2.contourArea(c) > 500]  # Mindestgröße anpassen falls nötig

        if not big_contours:
            msg = f"{file}: ❌ NO Leaf"
            print(msg)
            log_messages.append(msg)
            error_count += 1
            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()
            continue

        pixel_per_cm = scale

        # ================= DEBUG: MEASUREMENTS (für ALLE Blätter) =================
        debug_meas = warped.copy()
        leaf_number = 0  # Zähler für Blätter pro Bild

        for leaf in big_contours:
            leaf_number += 1
            hull = cv2.convexHull(leaf)
            area_px = cv2.contourArea(hull)
            area_cm2 = area_px / (pixel_per_cm**2)

            rect = cv2.minAreaRect(leaf)
            (w, h) = rect[1]
            angle = rect[2]

            # Normalisierung (Länge = längere Seite)
            if w < h:
                angle += 90
            length_cm = max(w, h) / pixel_per_cm
            width_cm = min(w, h) / pixel_per_cm

            # Debug: Blattkontur und Bounding Box zeichnen
            cv2.drawContours(debug_meas, [leaf], -1, (0, 255, 0), 2)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(debug_meas, [box], 0, (0, 0, 255), 2)
            cv2.putText(debug_meas, f"Leaf {leaf_number}: L={length_cm:.1f}cm, W={width_cm:.1f}cm",
                        (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            print(f"{file} | Leaf {leaf_number}: L={length_cm:.2f}cm, W={width_cm:.2f}cm, Area={area_cm2:.2f}cm², Angle={angle:.1f}°")
            results.append([file, leaf_number, length_cm, width_cm, area_cm2, angle])

        # Debug-Fenster anzeigen (skaliert)
        h_dbg, w_dbg = debug_meas.shape[:2]
        scale_dbg = min(800/w_dbg, 600/h_dbg)
        debug_resized = cv2.resize(debug_meas, (int(w_dbg*scale_dbg), int(h_dbg*scale_dbg)))
        cv2.imshow("DEBUG - All Leaves", debug_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # save CSV ==============================
    with open(output_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "leaf_number", "length_cm", "width_cm", "area_cm2", "angle_deg"])
        writer.writerows(results)

    # ================= SAVE LOG =================
    with open("log.txt", "w") as f:
        for line in log_messages:
            f.write(line + "\n")

    status_var.set(f"Ready | Mistakes: {error_count} | Total: {total_images}")

# ================= GUI labeling =================

root = ttk.Window(themename="solar")

root.title("Twinkels Leaf")
root.geometry("900x1350")

folder_var = ttk.StringVar()
output_var = ttk.StringVar()
status_var = ttk.StringVar()

# Description text
description = """
Leaf measurement using ArUco markers and OpenCV!

A white background with 4 ArUco markers is required for the measurement!
The markers must be positioned exactly 60 x 30 cm apart at their outermost corners.

This programme measures the length of leaves (narrow ones) and their width.
It calculates the leaf area based on the green colour.
An angle is displayed in the CSV file; this indicates the quality of the image.
It is recommended to scan the leaf as vertically as possible; other green 
objects should be covered.
The results are then saved in a CSV file.

For go next in the Analyzes press any Button!!


Created by Fledermausmann - C3D2
"""

ttk.Label(root,text="Leaf Sice Analyse (ArUco)",font=("Arial",12,"bold")).pack(pady=5)

ttk.Label(root,text=description,justify="left",wraplength=700).pack(pady=5)

ttk.Entry(root,textvariable=folder_var,width=50).pack()

ttk.Button(root,text="choose Picture-Folder",command=choose_folder).pack(pady=5)

ttk.Entry(root,textvariable=output_var,width=50).pack()

ttk.Button(root,text="CSV choose",command=choose_output).pack(pady=5)

ttk.Button(root,text="Start",command=start_analysis).pack(pady=10)

progress = ttk.Progressbar(root,length=300)
progress.pack(pady=5)

ttk.Label(root,textvariable=status_var).pack()

root.mainloop()