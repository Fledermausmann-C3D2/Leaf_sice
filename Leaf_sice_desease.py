import os
import cv2
import numpy as np
import csv
import re
import threading
import ttkbootstrap as ttk
from tkinter import filedialog

import hsv_tester


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

# ================= HSV Tester oeffnen =================
def open_hsv_tester():
    """Oeffnet den Dateidialog und startet den HSV-Tester in einem Thread."""
    image_path = filedialog.askopenfilename(
        title="Choose an image for HSV testing",
        filetypes=[
            ("JPEG", "*.jpg *.JPG *.jpeg *.JPEG"),
            ("PNG", "*.png *.PNG"),
            ("All files", "*.*"),
        ]
    )
    if not image_path:
        return
    thread = threading.Thread(target=hsv_tester.main, args=(image_path,))
    thread.start()
    status_var.set("HSV Tester started")

# ================= HSV aus GUI lesen =================
def get_hsv(name):
    """Liest die HSV-Grenzen fuer eine Farbvariante aus den GUI-Eingaben."""
    lower = [int(v.get()) for v in hsv_vars[name]["lower"]]
    upper = [int(v.get()) for v in hsv_vars[name]["upper"]]
    return np.array(lower), np.array(upper)

def clamp_to_main(lower_sub, upper_sub, lower_main, upper_main):
    """Begrenzt die HSV-Werte eines Unterbereichs so, dass sie
    automatisch innerhalb des main-Bereichs liegen."""
    lower_sub = np.maximum(lower_sub, lower_main)
    upper_sub = np.minimum(upper_sub, upper_main)
    return lower_sub, upper_sub

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
        pixel_per_cm = scale
        try:
            real_w = float(marker_width_var.get())
        except ValueError:
            real_w = REAL_WIDTH_CM
        try:
            real_h = float(marker_height_var.get())
        except ValueError:
            real_h = REAL_HEIGHT_CM
        dst = np.array([
            [0,0],
            [real_w*scale,0],
            [real_w*scale,real_h*scale],
            [0,real_h*scale]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (int(real_w*scale), int(real_h*scale)))
    
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        debug_meas = warped.copy() 
        leaf_number = 0


      # ================= LEAF COLOR SEGMENTATION =================

        # Sättigungsmaske
        mask_sat = (hsv[:,:,1] > 30).astype(np.uint8) * 255


        # ================= ALLES GRÜN =================
        # Dieser Bereich ist die Basis = 100 %

        lower_green, upper_green = get_hsv("main")

        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Sättigung berücksichtigen
        mask_green = cv2.bitwise_and(mask_green, mask_sat)


        # ================= UNTERBEREICHE =================
        # Die Grenzen von light / medium / dark werden automatisch
        # auf den main-Bereich begrenzt, sodass sie immer innerhalb
        # von main liegen.

        # HELLGRÜN
        lower_light_green, upper_light_green = get_hsv("hell")
        lower_light_green, upper_light_green = clamp_to_main(
            lower_light_green, upper_light_green, lower_green, upper_green)

        # MITTELGRÜN
        lower_medium_green, upper_medium_green = get_hsv("mittel")
        lower_medium_green, upper_medium_green = clamp_to_main(
            lower_medium_green, upper_medium_green, lower_green, upper_green)

        # DUNKELGRÜN
        lower_dark_green, upper_dark_green = get_hsv("dunkel")
        lower_dark_green, upper_dark_green = clamp_to_main(
            lower_dark_green, upper_dark_green, lower_green, upper_green)


        # Masken erstellen
        mask_light = cv2.inRange(hsv, lower_light_green, upper_light_green)
        mask_medium = cv2.inRange(hsv, lower_medium_green, upper_medium_green)
        mask_dark = cv2.inRange(hsv, lower_dark_green, upper_dark_green)


        # Nur Pixel behalten, die auch in "ALLES GRÜN" liegen
        mask_light = cv2.bitwise_and(mask_light, mask_green)
        mask_medium = cv2.bitwise_and(mask_medium, mask_green)
        mask_dark = cv2.bitwise_and(mask_dark, mask_green)


        # Morphologische Operationen
        kernel = np.ones((7,3), np.uint8)

        mask_green = cv2.morphologyEx(
            mask_green, cv2.MORPH_CLOSE, kernel
        )

        mask_light = cv2.morphologyEx(
            mask_light, cv2.MORPH_CLOSE, kernel
        )

        mask_medium = cv2.morphologyEx(
            mask_medium, cv2.MORPH_CLOSE, kernel
        )

        mask_dark = cv2.morphologyEx(
            mask_dark, cv2.MORPH_CLOSE, kernel
        )


        # ================= GESAMTBLATT =================
        # Für die Blatterkennung wird ALLES GRÜN verwendet.

        mask_total = mask_green


        # Konturen finden
        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big_contours = [c for c in contours if cv2.contourArea(c) > 500]

        if not big_contours:
            msg = f"{file}: ❌ NO Leaf"
            print(msg)
            log_messages.append(msg)
            error_count += 1
            status_var.set(f"{i+1}/{total_images} | {msg}")
            root.update_idletasks()
            continue

        # Flächen pro Blatt und Farbton berechnen + Debug-Ausgabe
        for leaf in big_contours:
            leaf_number += 1

            
            hull = cv2.convexHull(leaf)
            area_px_total = cv2.contourArea(hull)
            area_cm2_total = area_px_total / (pixel_per_cm**2)

            # Maske für das aktuelle Blatt
            leaf_mask = np.zeros_like(mask_total)
            cv2.drawContours(leaf_mask, [hull], -1, 255, -1)

            # Masken für Farbtöne auf Blatt begrenzen
            leaf_mask_light = cv2.bitwise_and(mask_light, leaf_mask)
            leaf_mask_medium = cv2.bitwise_and(mask_medium, leaf_mask)
            leaf_mask_dark = cv2.bitwise_and(mask_dark, leaf_mask)

            # Pixel pro Farbton zählen
            area_px_light = cv2.countNonZero(leaf_mask_light)
            area_px_medium = cv2.countNonZero(leaf_mask_medium)
            area_px_dark = cv2.countNonZero(leaf_mask_dark)

            # Flächen in cm²
            area_cm2_light = area_px_light / (pixel_per_cm**2)
            area_cm2_medium = area_px_medium / (pixel_per_cm**2)
            area_cm2_dark = area_px_dark / (pixel_per_cm**2)

            # ================= PROZENTUALE FARBVERTEILUNG =================
            # Alle erkannten Grüntöne zusammen = 100 %

            area_px_green = cv2.countNonZero(
                cv2.bitwise_and(mask_green, leaf_mask)
            )


            if area_px_green > 0:
                percent_light = (area_px_light / area_px_green) * 100
                percent_medium = (area_px_medium / area_px_green) * 100
                percent_dark = (area_px_dark / area_px_green) * 100
            else:
                percent_light = 0
                percent_medium = 0
                percent_dark = 0

            # Gesamte erkannte Grünfläche in cm²
            area_cm2_green = area_px_green / (pixel_per_cm**2)



            # Debug: Blattkontur und Bounding Box zeichnen
            cv2.drawContours(debug_meas, [leaf], -1, (0, 255, 0), 2)
            rect = cv2.minAreaRect(leaf)
            (w, h) = rect[1]
            angle = rect[2]
            if w < h:
                angle += 90
            length_cm = max(w, h) / pixel_per_cm
            width_cm = min(w, h) / pixel_per_cm

# Print-Anweisung für die Konsole
            print(f"{file} | Leaf {leaf_number}: L={length_cm:.2f}cm, W={width_cm:.2f}cm, Area={area_cm2_total:.2f}cm², Angle={angle:.1f}°")
            print(f"  Light: {area_cm2_light:.2f}cm² ({percent_light:.1f}%), Medium: {area_cm2_medium:.2f}cm² ({percent_medium:.1f}%), Dark: {area_cm2_dark:.2f}cm² ({percent_dark:.1f}%)")



            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(debug_meas, [box], 0, (0, 0, 255), 2)
            cv2.putText(debug_meas, f"Leaf {leaf_number}: L={length_cm:.1f}cm, W={width_cm:.1f}cm, Total={area_cm2_total:.1f}cm²",
                        (box[0][0], box[0][1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(debug_meas, f"Light: {area_cm2_light:.1f}cm² ({percent_light:.1f}%)",
                        (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 2)
            cv2.putText(debug_meas, f"Medium: {area_cm2_medium:.1f}cm² ({percent_medium:.1f}%)",
                        (box[0][0], box[0][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 2)
            cv2.putText(debug_meas, f"Dark: {area_cm2_dark:.1f}cm² ({percent_dark:.1f}%)",
                        (box[0][0], box[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 200), 2)

            # Ergebnisse speichern (kombiniert)
            results.append([
                file,
                leaf_number,
                length_cm,
                width_cm,
                area_cm2_total,
                area_cm2_green,
                area_cm2_light,
                percent_light,
                area_cm2_medium,
                percent_medium,
                area_cm2_dark,
                percent_dark,
                angle
            ])


        # Debug-Fenster anzeigen (skaliert)
        h_dbg, w_dbg = debug_meas.shape[:2]
        scale_dbg = min(800/w_dbg, 600/h_dbg)
        debug_resized = cv2.resize(debug_meas, (int(w_dbg*scale_dbg), int(h_dbg*scale_dbg)))
        cv2.imshow("DEBUG - All Leaves", debug_resized)
        #cv2.imshow("Light Green", mask_light)
        #cv2.imshow("Medium Green", mask_medium)
        #cv2.imshow("Dark Green", mask_dark)

        # 10 Sekunden warten oder auf Tastendruck
        start_time = cv2.getTickCount()
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key != 255:  # Taste gedrückt
                break
            current_time = cv2.getTickCount()
            elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
            if elapsed_time >= 1:  # 10 Sekunden vergangen
                break
        cv2.destroyAllWindows()

    # save CSV ==============================
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "leaf_number",
            "length_cm",
            "width_cm",
            "area_total_cm2",
            "area_green_cm2",
            "area_light_cm2",
            "percent_light",
            "area_medium_cm2",
            "percent_medium",
            "area_dark_cm2",
            "percent_dark",
            "angle_deg"
        ])

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
marker_width_var = ttk.StringVar(value=str(REAL_WIDTH_CM))
marker_height_var = ttk.StringVar(value=str(REAL_HEIGHT_CM))

# ================= HSV-Variablen (alle 4 Varianten editierbar) =================
hsv_defaults = {
    "main":   {"lower": [25, 30, 50],  "upper": [90, 255, 255]},
    "hell":   {"lower": [25, 30, 180], "upper": [90, 255, 255]},
    "mittel": {"lower": [25, 50, 100], "upper": [90, 255, 180]},
    "dunkel": {"lower": [25, 50, 50],  "upper": [90, 255, 100]},
}

hsv_vars = {}
for _name, _bounds in hsv_defaults.items():
    hsv_vars[_name] = {}
    for _bound in ("lower", "upper"):
        hsv_vars[_name][_bound] = [
            ttk.StringVar(value=str(_bounds[_bound][0])),
            ttk.StringVar(value=str(_bounds[_bound][1])),
            ttk.StringVar(value=str(_bounds[_bound][2])),
        ]

# Description text
description = """
Leaf disease detection using ArUco markers and OpenCV!

This programme identifies infected leaves and quantifies the dark and light
areas on each leaf as a percentage, using adjustable HSV colour thresholds.

A white background with 4 ArUco markers is required for the measurement.
The markers must be positioned exactly 60 x 30 cm apart at their outermost
corners. The programme uses them to correct the perspective and to determine
the real size of each leaf in centimetres.

The HSV settings define four colour ranges:
 - main   = the main green range. This defines the total leaf area (100%).
            Only pixels within this range are considered part of the leaf.
 - light  = light spots within the leaf. The programme counts how many of
            the main pixels fall into this lighter range and reports it as
            a percentage of the total leaf area.
 - medium = medium-coloured spots within the leaf, reported as a percentage.
 - dark   = dark spots within the leaf, reported as a percentage.

The light, medium and dark ranges are automatically clamped so that they
always lie within the main range. You only need to adjust main to capture the
whole leaf; the three sub-ranges can then be fine-tuned to pick out the light,
medium and dark regions you want to quantify.

For each leaf the following values are saved in a CSV file:
 image, leaf number, length (cm), width (cm), total area (cm2),
 green area (cm2), light area + percent, medium area + percent,
 dark area + percent, and an angle describing the image quality.

It is recommended to scan the leaf as flat and vertically as possible and
to cover any other green objects in the frame.

For go next in the Analyzes press any Button!!


Created by Fledermausmann - C3D2
"""

ttk.Label(root,text="Leaf Sice Analyse (ArUco)",font=("Arial",12,"bold")).pack(pady=5)

ttk.Label(root,text=description,justify="left",wraplength=700).pack(pady=5)

ttk.Entry(root,textvariable=folder_var,width=50).pack()

ttk.Button(root,text="choose Picture-Folder",command=choose_folder).pack(pady=5)

ttk.Entry(root,textvariable=output_var,width=50).pack()

ttk.Button(root,text="CSV choose",command=choose_output).pack(pady=5)

ttk.Label(root,text="Marker distance (cm):",
          font=("Arial",10,"bold")).pack(pady=(10,0))
marker_frame = ttk.Frame(root)
marker_frame.pack(pady=2)
ttk.Label(marker_frame, text="width:", width=8).pack(side="left")
ttk.Entry(marker_frame, textvariable=marker_width_var, width=8).pack(side="left", padx=2)
ttk.Label(marker_frame, text="height:", width=8).pack(side="left")
ttk.Entry(marker_frame, textvariable=marker_height_var, width=8).pack(side="left", padx=2)

ttk.Label(root,text="HSV settings  (lower H,S,V  /  upper H,S,V):",
          font=("Arial",10,"bold")).pack(pady=(10,0))

def build_hsv_row(name, label, highlight=False):
    frame = ttk.Frame(root)
    frame.pack(pady=2)
    label_font = ("Arial", 10, "bold") if highlight else ("Arial", 10)
    ttk.Label(frame, text=label, width=8, font=label_font,
              bootstyle="primary" if highlight else "default").pack(side="left")
    ttk.Label(frame, text="lower:", width=6).pack(side="left")
    for v in hsv_vars[name]["lower"]:
        ttk.Entry(frame, textvariable=v, width=4).pack(side="left", padx=1)
    ttk.Label(frame, text="upper:", width=6).pack(side="left")
    for v in hsv_vars[name]["upper"]:
        ttk.Entry(frame, textvariable=v, width=4).pack(side="left", padx=1)

build_hsv_row("main",   "main",   highlight=True)
build_hsv_row("hell",   "light")
build_hsv_row("mittel", "medium")
build_hsv_row("dunkel", "dark")

ttk.Button(root,text="HSV color test",command=open_hsv_tester).pack(pady=5)

ttk.Button(root,text="Start",command=start_analysis).pack(pady=10)

progress = ttk.Progressbar(root,length=300)
progress.pack(pady=5)

ttk.Label(root,textvariable=status_var).pack()

root.mainloop()