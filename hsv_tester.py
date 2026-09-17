"""HSV color tester - load an image and find the right HSV range interactively.

Run standalone or open it from the main programme via the "HSV color test" button.
"""
import cv2
import numpy as np
import sys
import os


def nothing(x):
    pass


def main(image_path=None):
    # Bildpfad als Argument, uebergeben oder Dateiauswahl
    if image_path is None:
        if len(sys.argv) > 1:
            image_path = sys.argv[1]
        else:
            from tkinter import Tk, filedialog
            Tk().withdraw()
            image_path = filedialog.askopenfilename(
                title="Choose an image for HSV testing",
                filetypes=[
                    ("JPEG", "*.jpg *.JPG *.jpeg *.JPEG"),
                    ("PNG", "*.png *.PNG"),
                    ("All files", "*.*"),
                ]
            )

    if not image_path or not os.path.exists(image_path):
        print("No image selected.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image: {image_path}")
        return

    # Arbeitskopie skalieren, damit das Fenster auf den Bildschirm passt.
    h, w = img.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ih, iw = img.shape[:2]

    cv2.namedWindow("HSV Tester", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow("HSV Tester", 900, 700)
    cv2.moveWindow("HSV Tester", 10, 10)

    # Trackbars: lower H,S,V und upper H,S,V
    cv2.createTrackbar("Low-H", "HSV Tester", 25, 179, nothing)
    cv2.createTrackbar("Low-S", "HSV Tester", 30, 255, nothing)
    cv2.createTrackbar("Low-V", "HSV Tester", 50, 255, nothing)
    cv2.createTrackbar("Up-H", "HSV Tester", 90, 179, nothing)
    cv2.createTrackbar("Up-S", "HSV Tester", 255, 255, nothing)
    cv2.createTrackbar("Up-V", "HSV Tester", 255, 255, nothing)

    # Klick-Picker: HSV-Wert des angeklickten Pixels anzeigen.
    # Das Fenster zeigt das skalierte 'show'-Bild; die Mauskoordinaten
    # liegen daher in show-Pixeln und werden ueber 'view["f"]' zurueck
    # in combined/Original-Pixel gerechnet.
    click = {"x": None, "y": None, "h": None, "s": None, "v": None}
    view = {"f": 1.0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            f = view["f"] or 1.0
            px = int(x / f)
            py = int(y / f)
            if 0 <= px < iw and 0 <= py < ih:
                hh, ss, vv = hsv[py, px]
                click["x"], click["y"] = px, py
                click["h"], click["s"], click["v"] = int(hh), int(ss), int(vv)
                print(f"HSV @ ({px},{py}) = H={hh} S={ss} V={vv}")

    cv2.setMouseCallback("HSV Tester", on_mouse)

    print("HSV Tester - adjust the trackbars to find your range.")
    print("Click on the left image to read the HSV value of a pixel.")
    print("Press 's' to print the current values, 'q' or ESC to quit.")

    while True:
        l_h = cv2.getTrackbarPos("Low-H", "HSV Tester")
        l_s = cv2.getTrackbarPos("Low-S", "HSV Tester")
        l_v = cv2.getTrackbarPos("Low-V", "HSV Tester")
        u_h = cv2.getTrackbarPos("Up-H", "HSV Tester")
        u_s = cv2.getTrackbarPos("Up-S", "HSV Tester")
        u_v = cv2.getTrackbarPos("Up-V", "HSV Tester")

        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        # Maske neben Ergebnis legen
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((img, mask_bgr, result))

        # Klick-Marker ins Original-Pane zeichnen (ohne Text, Text kommt
        # in die Statusleiste unten, damit es immer gut lesbar ist).
        if click["x"] is not None:
            cx, cy = click["x"], click["y"]
            cv2.drawMarker(combined, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # Statusleiste unten: schwarzer Balken mit weißer Schrift,
        # zeigt den HSV-Wert des letzten Klicks und die Trackbar-Werte.
        bar_h = 40
        bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
        if click["x"] is not None:
            info = (f"Clicked HSV: H={click['h']} S={click['s']} V={click['v']}    "
                   f"Range: lower=[{l_h},{l_s},{l_v}] upper=[{u_h},{u_s},{u_v}]")
        else:
            info = (f"No click yet - click on the left image    "
                   f"Range: lower=[{l_h},{l_s},{l_v}] upper=[{u_h},{u_s},{u_v}]")
        cv2.putText(bar, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        combined = np.vstack((combined, bar))

        # Auf eine Anzeigebreite begrenzen, damit das Fenster auf den
        # Bildschirm passt (sonst ragt es unter Windows heraus).
        max_show_w = 1200
        show = combined
        f = 1.0
        if combined.shape[1] > max_show_w:
            f = max_show_w / combined.shape[1]
            show = cv2.resize(combined, (int(combined.shape[1] * f),
                                         int(combined.shape[0] * f)))
        view["f"] = f

        cv2.imshow("HSV Tester", show)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            print(f"lower = [{l_h}, {l_s}, {l_v}]  upper = [{u_h}, {u_s}, {u_v}]")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
