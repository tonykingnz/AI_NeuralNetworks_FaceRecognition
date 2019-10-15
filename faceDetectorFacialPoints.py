import cv2
import dlib

def imprimePontos(imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
webcam = cv2.VideoCapture(0)
while True:
    _, imagem = webcam.read()
    detectorFace = dlib.get_frontal_face_detector()
    detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
    facesDetectadas = detectorFace(imagem)
    for face in facesDetectadas:
        pontos = detectorPontos(imagem, face)
        print(pontos.parts())
        print(len(pontos.parts()))
        imprimePontos(imagem, pontos)

    cv2.imshow("Pontos Faciais", imagem)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()