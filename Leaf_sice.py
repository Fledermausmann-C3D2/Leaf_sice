import os
import cv2
import numpy as np
import csv
import threading
import ttkbootstrap as ttk
from tkinter import filedialog


#================== Feler loggen============
log_messages = []
error_count = 0

# ================= Reale Größe 4 Marker =================
REAL_WIDTH_CM = 30
REAL_HEIGHT_CM = 60

# ================= GUI FUNKTIONEN =================

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

    # ArUco Setup
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
   
    parameters = aruco.DetectorParameters()
    #ggf feinere erkennung der Marker -->

    #parameters.adaptiveThreshWinSizeMin = 3
    #parameters.adaptiveThreshWinSizeMax = 23
    #parameters.adaptiveThreshWinSizeStep = 10
    #parameters.adaptiveThreshConstant = 7 #(rauschen)

    image_paths = []

    for root_dir, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith((".jpg",".png",".jpeg")):
                image_paths.append(os.path.join(root_dir,file))

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

        if ids is None or len(corners) < 4:
            #print(file, "❌ Marker fehlen")

            msg = f"{file}: ❌ Marker missing"

            print(msg)
            log_messages.append(msg)
            error_count += 1

            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()

            continue

        # alle Punkte sammeln
        pts_all = []
        for c in corners:
            for p in c[0]:
                pts_all.append(p)

        pts_all = np.array(pts_all)

        # äußere Form bestimmen
        hull = cv2.convexHull(pts_all)

        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) != 4:
            #print(file, "❌ keine 4 Ecken")
            msg = f"{file}: ❌ no 4 corners"

            print(msg)
            log_messages.append(msg)
            error_count += 1

            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()

            continue

        rect = order_points(approx.reshape(4,2))

        # Zielsystem
        scale = 20 #skalierung der Flächenberechnung 10 - 30
        dst = np.array([
            [0,0],
            [REAL_WIDTH_CM*scale,0],
            [REAL_WIDTH_CM*scale,REAL_HEIGHT_CM*scale],
            [0,REAL_HEIGHT_CM*scale]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(original, M, (int(REAL_WIDTH_CM*scale), int(REAL_HEIGHT_CM*scale)))

        # ================= BLATT =================
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

        # Großer farbbereich auch für leicht gelbe blätter
        # sonst 25,40,40 - 90,255,255
        lower_green = np.array([25,40,40])
        upper_green = np.array([90,255,255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Morphologie einstellen (3,3)kleines blatt (7,7)großes
        # 3,1 weil schmal
        kernel = np.ones((25,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            #print(file, "❌ kein Blatt")

            msg = f"{file}: ❌ NO Leaf"

            print(msg)
            log_messages.append(msg)
            error_count += 1

            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()
            continue

        leaf = max(contours, key=cv2.contourArea)
      

        # ================= Maße =================
        pixel_per_cm = scale

        #bessert fehler, einkerbungen aus hull
        hull = cv2.convexHull(leaf)
        area_px = cv2.contourArea(hull)

        area_cm2 = area_px / (pixel_per_cm**2)
        
        #misst anhand von fester achse============
        #x,y,w,h = cv2.boundingRect(leaf)

        #width_cm = w / pixel_per_cm
        #height_cm = h / pixel_per_cm

        #misst mit rotation, nach blatt===========
        rect = cv2.minAreaRect(leaf)
        (w, h) = rect[1]
        angle = rect[2]

        # Normalisierung ============================
        if w < h:
            angle = angle + 90

        length_cm = max(w, h) / pixel_per_cm
        width_cm  = min(w, h) / pixel_per_cm



        print(file, length_cm, width_cm, area_cm2, angle)

        results.append([file, length_cm, width_cm, area_cm2, angle])

        progress["value"] = i + 1
        #status_var.set(f"{i+1}/{total_images}")
        status_var.set(f"{i+1}/{total_images} | OK: {file}")
        root.update_idletasks()

    # CSV speichern ==============================
    with open(output_csv,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image","length_cm","width_cm","area_cm2","angle_deg"])
        writer.writerows(results)

    # ================= LOG SPEICHERN =================
    with open("log.txt", "w") as f:
        for line in log_messages:
            f.write(line + "\n")

    #status_var.set("Fertig")
    status_var.set(f"Ready | Mistakes: {error_count} | Total: {total_images}")

# ================= GUI beschriftung =================

root = ttk.Window(themename="solar")

root.title("Twinkels Leaf")
root.geometry("900x1350")

folder_var = ttk.StringVar()
output_var = ttk.StringVar()
status_var = ttk.StringVar()

# Erklärungstext
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

Translated with DeepL.com (free version)

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