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

    # Skalieren, falls zu gross
    h, w = img.shape[:2]
    max_dim = 900
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.namedWindow("HSV Tester", cv2.WINDOW_NORMAL)

    # Trackbars: lower H,S,V und upper H,S,V
    cv2.createTrackbar("Low-H", "HSV Tester", 25, 179, nothing)
    cv2.createTrackbar("Low-S", "HSV Tester", 30, 255, nothing)
    cv2.createTrackbar("Low-V", "HSV Tester", 50, 255, nothing)
    cv2.createTrackbar("Up-H", "HSV Tester", 90, 179, nothing)
    cv2.createTrackbar("Up-S", "HSV Tester", 255, 255, nothing)
    cv2.createTrackbar("Up-V", "HSV Tester", 255, 255, nothing)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print("HSV Tester - adjust the trackbars to find your range.")
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

        cv2.imshow("HSV Tester", combined)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            print(f"lower = [{l_h}, {l_s}, {l_v}]  upper = [{u_h}, {u_s}, {u_v}]")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
