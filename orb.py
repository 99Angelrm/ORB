import cv2 as cv   #importar la librería de open CV
gray1 =cv.imread ('Foto_guicho.jpg', cv.IMREAD_GRAYSCALE) #Carga de la foto en escala de grises
cap = cv.VideoCapture(0) #inicio de la captura de imágenes
orb = cv.ORB_create() #crea el objeto orb
kpl, des1 = orb.detectAndCompute(gray1, None) #Encuentra keypoints y descriptors with ORB
while 1: #Ciclo para seguir obteniendo imagenes de la cámara
    ret, frame = cap.read() #lee la entrada de la camara
    
    gray2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Obtenemos el frame y lo pasamos a escala de grises 
    kp2, des2 = orb.detectAndCompute (gray2, None) # Encuentra keypoints y descriptors with ORB

    brute_force_matching = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  #Herramienta para los "match"

    matches = brute_force_matching.match(des1, des2)  #"Match" descriptors
    matches = sorted (matches, key= lambda x:x.distance) #Los ordena con respespecto a la distancia

    matching_result= cv.drawMatches(gray1, kpl, gray2, kp2, matches[:20], None) # Dibuja los "matches"

    cv.imshow("Original GrayScale Image",gray1) # Imprime la imagen original
    cv.imshow ("Printed Grayscale Image", gray2) # Imprime el video de la cámara
    cv.imshow("Matching Result.png",matching_result) #   Imprime los matches

    if cv.waitKey(1) & 0xFF == ord('q'): # Espera si oprimes la tecla q
        cap.release() #   Detiene el video
        cv.destroyAllWindows() # Destruye todas las ventanas
        break


