import cv2
import dlib

webcam = cv2.VideoCapture(0)
while True:
    s, imagem = webcam.read()
    detector = dlib.get_frontal_face_detector()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detector(imagemCinza)
    for faces in facesDetectadas:
        e, t, d, b = (int(faces.left()), int(faces.top()), int(faces.right()), int(faces.bottom()))
        cv2.rectangle(imagem, (e, t), (d,b), (0,255,0), 2)
    cv2.imshow("Faces Detectadas", imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


webcam.release()
cv2.destroyAllWindows()