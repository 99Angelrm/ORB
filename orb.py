import cv2 as cv
import numpy as np
gray1 =cv.imread ('lena_std.png', cv.IMREAD_GRAYSCALE) 
gray2 = cv.imread ('../Miaus.png', cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()

kpl, des1 = orb.detectAndCompute(gray1, None) 
kp2, des2 = orb.detectAndCompute (gray2, None)

brute_force_matching = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

matches = brute_force_matching.match(des1, des2) 
matches = sorted (matches, key= lambda x:x.distance)

matching_result= cv.drawatches(gray1, kpl, gray2, kp2, matches[:20], None)

cv.imshow("Original GrayScale Image",gray1) 
cv.imshow ("Printed Grayscale Image", gray2) 
cv.imshow("Matching Result.png",matching_result)

cv.waitKey(0)
cv.destroyAllWindows()
