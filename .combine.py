import cv2
import numpy as np

img = cv2.imread("../historical_function/img_crop_1_1.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

img_mop = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
img_mop = cv2.morphologyEx(img_mop, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))

revert = img_mop
for i in range(0, revert.shape[0]):
    for j in range(0, revert.shape[1]):
        px = 255 if revert[i][j] == 0 else 0
        revert[i][j] = px
cv2.imwrite("revert.png", revert)
canny = cv2.Canny(revert, 100, 200)

kontur1, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, kontur1, -1, (0, 255, 0), 5)

img_revert = cv2.imread("revert.png")
revert_gray = cv2.cvtColor(img_revert, cv2.COLOR_BGR2GRAY)
threshold_inv = cv2.threshold(revert_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = np.ones((3, 3), np.uint8)
mop_open = cv2.morphologyEx(threshold_inv, cv2.MORPH_OPEN, kernel, iterations=2)

dilate = cv2.dilate(mop_open, kernel, iterations=2)

distance_trans = cv2.distanceTransform(mop_open, cv2.DIST_L2, cv2.DIST_MASK_5)
dist_thres = cv2.threshold(distance_trans, 0.24 * distance_trans.max(), 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("distace.png", dist_thres)

img_distace = cv2.imread("distace.png", 0)
kontur2, _ = cv2.findContours(img_distace, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# mc = 264.5833  # 1 px = 264.5833 micrometer

for i in range(0, len(kontur2)):
    ((x, y), r) = cv2.minEnclosingCircle(kontur2[i])
    wide = cv2.contourArea(kontur2[i], False)
    if wide == 0: continue
    cv2.putText(img, f"{int(wide)}px", (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)
    # cv2.putText(img, f"{int(wide * mc)}μm", (int(x) - 4, int(y)), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 255), 1)


cv2.imshow('aa', img)
cv2.waitKey()