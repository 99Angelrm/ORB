import cv2 as cv
gray1 =cv.imread ('angel_photo.jpg', cv.IMREAD_GRAYSCALE)
cap = cv.VideoCapture(0)
orb = cv.ORB_create()
kpl, des1 = orb.detectAndCompute(gray1, None)
while 1:
    ret, frame = cap.read()
    
    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute (gray2, None)

    brute_force_matching = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = brute_force_matching.match(des1, des2) 
    matches = sorted (matches, key= lambda x:x.distance)

    matching_result= cv.drawMatches(gray1, kpl, gray2, kp2, matches[:20], None)

    cv.imshow("Original GrayScale Image",gray1) 
    cv.imshow ("Printed Grayscale Image", gray2) 
    cv.imshow("Matching Result.png",matching_result)   

    if cv.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv.destroyAllWindows()  
        break


