import cv2
import numpy as np
from keras.models import load_model

#################### CARGA DEL MODELO DE CLASIFICACIÓN DE LENGUA DE SIGNOS:
# TO DO
# Tocará cambiar el modelo por el modelo final que usemos
# Cargamos el modelo de clasificación de lengua de signos
model = load_model(filepath='Imagenes/Pruebas/prueba3Andres.h5', compile=False)
# Cargamos el diccionario de signos de la lengua de signos
sign_language_dict = np.load('Imagenes/Pruebas/i2c.npy', allow_pickle=True).item()

#################### DEFINICIÓN DEL DISPOSITIVO DE CAPTURA DE VIDEO:
# Establecemos el dispositivo de captura de video -> 0 = webcam
cap = cv2.VideoCapture(0)

#################### DEFINICIÓN DE LA REGIÓN DE INTERÉS (ROI):
# Definimos la región de interest (ROI, por sus siglas en inglés)
# Es la zona de la imagen donde queremos que el usuario sitúe la mano
# La mano se posicionará en la parte derecha medio-superior de la imagen
# NOTE
# Puede ser necesario ajustar estos valores en función de la cámara utilizada 
# y de la distancia y posición de la mano del usuario
mano = input("Elige mano derecha o mano izquierda(D/I): ")
if mano == "D":
    top, right, bottom, left = 100, 600, 300, 400
elif mano == "I":
    top, right, bottom, left = 100, 300, 300, 100

#################### BUCLE PRINCIPAL DE CAPTURA DE VIDEO:
# Empezamos a capturar fotogramas del dispositivo de captura de video
while cap.isOpened():
    # Leemos el frame desde el dispositivo de captura de video
    ret, frame = cap.read()

    # Si el frame se ha leído correctamente, continuamos
    if ret:
    
        # Volteamos el frame horizontalmente para tener un efecto espejo
        frame = cv2.flip(frame, 1)

        # Dibujamos el rectángulo delimitador alrededor donde el usuario deberá poner la mano
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Recortamos el frame para quedarnos con la ROI -> esta será la imagen de la mano
        roi = frame[top:bottom, left:right]
        #################### PREDICCIÓN DEL SIGNO DE LA LENGUA DE SIGNOS:
        # Convertimos la imagen de la ROI a escala de grises 
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Preprocesamos la imagen de la mano para el modelo de clasificación de lengua de signos
        # 1) Redimensionamos la imagen de la mano a 28x28 píxeles
        hand_image = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        # 2) La normalizamos para que los valores de los píxeles estén entre 0 y 1
        hand_image = hand_image / 255
        # 3) Añadimos la dimensión del canal en el último eje para que la imagen tenga la forma (28, 28, 1)
        hand_image = np.expand_dims(hand_image, axis=-1)
        # 4) Añadimos la dimensión del batch para que la imagen tenga la forma (1, 28, 28, 1)
        hand_image = np.expand_dims(hand_image, axis=0)
    
        # Pasamos la imagen por el modelo y hacemos la predicción
        prediction = model(hand_image, training=False)
        predicted_sign = np.argmax(prediction, axis=1)[0]
        certainty = prediction[0][predicted_sign]*100
        predicted_sign = sign_language_dict[predicted_sign]            
        # Escribimos el signo predicho encima del rectángulo delimitador de la mano
        # (frame, texto, posición, fuente, tamaño, color, grosor)
        cv2.putText(frame, f'{predicted_sign}: {certainty:.2f}%', (left+10, bottom-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #Añadir que si pulso espacio aparezca imagenes de la LSA o LSE

        # Mostramos el frame capturado actualizado
        cv2.imshow('Interprete artificial de lengua de signos', frame)

        if cv2.waitKey(1) == 32:
            image = cv2.imread('Imagenes\SignosLSA.png')
            cv2.imshow('Abecedario LSA', image)
        

        # Comprobamos si el usuario quiere salir del programa (vemos si ha pulsado la tecla ENTER)
        if cv2.waitKey(1) == 13:
            cv2.destroyAllWindows()
            break
    
    # Si no se ha podido leer el frame, lanzamos una excepción
    else:
        raise RuntimeError('Error al leer de la cámara.')

# Liberamos el dispositivo de captura de vídeo y cerramos todas las ventanas
cap.release()
cv2.destroyAllWindows()