import cv2

webcam = cv2.VideoCapture(0)
while True:
    s,imagem = webcam.read()
    imagem = cv2.flip(imagem, 180)
    classificador = cv2.CascadeClassifier("recursos/haarcascade_frontalface_default.xml")
    facesDetectadas = classificador.detectMultiScale(imagem)
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)

        cv2.imshow("Detector haar", imagem)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

webcam.release()
cv2.destroyAllWindows()


