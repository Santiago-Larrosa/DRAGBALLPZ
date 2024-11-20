import cv2
import mediapipe as mp
import math
import pygame
import time
import numpy as np 
# COSO DE LOS SONIDOS QUE NO SE SI FUNCIONA
pygame.mixer.init()

# PSOES DEL MEDIA
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# ARRANCA LA CAMARA
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# AGARRAR IMAGENES
ki_image = cv2.imread('ki_image.png', cv2.IMREAD_UNCHANGED) 
kamehameha_image = cv2.imread('kamehameha_image.png', cv2.IMREAD_UNCHANGED)  
makankosapo_image = cv2.imread('makankosapo.png')
masenko_image = cv2.imread('masenko.png', cv2.IMREAD_UNCHANGED)
# AGARRAR SONIDOS
ki_sound = pygame.mixer.Sound('ki_sound.wav') 
kamehameha_sound = pygame.mixer.Sound('kamehameha_sound.wav')
makankosapo_sound = pygame.mixer.Sound('makankosapo.wav')
masenko_sound = pygame.mixer.Sound('kamehameha_sound.wav')
ex_sound = pygame.mixer.Sound('explosion.wav')


# ABRO LA VENTANA
cv2.namedWindow('Detección de movimientos', cv2.WINDOW_NORMAL)

# CALCULO ANGULO DE BRAZO
def calcular_angulo(hombro, codo, muñeca):
    v1 = ((hombro.x - codo.x), (hombro.y - codo.y))
    v2 = ((muñeca.x - codo.x), (muñeca.y - codo.y))
    producto_escalar = v1[0] * v2[0] + v1[1] * v2[1]
    magnitud_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitud_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    cos_angulo = producto_escalar / (magnitud_v1 * magnitud_v2)
    angulo = math.degrees(math.acos(cos_angulo))
    return angulo

# FUNCION DE FORMULA DE CALCULO DE DISTANCIA VECTORIAL ENTRE DOS PUNTOS
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# PUÑO CERRADO
def comprobar_puño_cerrado(hand_landmarks, umbral_cerrado=0.05):
    dedos_cerrados = 0
    puntos_dedos = [4, 8, 12, 16, 20]
    puntos_nudillos = [2, 5, 9, 13, 17]
    
    for i in range(1, 5):  # PULGAR NO
        distancia = calcular_distancia(hand_landmarks.landmark[puntos_dedos[i]], hand_landmarks.landmark[puntos_nudillos[i]])
        if distancia < umbral_cerrado:
            dedos_cerrados += 1

    return dedos_cerrados >= 3

    # MANO ARRIBA, CINTURA SOLA, LA MEDIA VUELTA...
def comprobar_manos_abiertas(hand_landmarks, umbral_abierta=0.1):
    dedos_abiertos = 0
    puntos_dedos = [8, 12, 16, 20]  # PUNTA DE LOS DEDOS
    puntos_nudillos = [6, 10, 14, 18]  # EL RESTO DEL DEDO XD

    for i in range(len(puntos_dedos)):
        distancia = calcular_distancia(hand_landmarks.landmark[puntos_dedos[i]], hand_landmarks.landmark[puntos_nudillos[i]])
        if distancia > umbral_abierta:  # DEDO ESTIRADO
            dedos_abiertos += 1

    return dedos_abiertos >= 3

# KAMEHAMEHA FUNCTION (GOKU BASE >>>>>>>>>>> SAITAMA)
def manos_juntas(landmark_izq, landmark_der):
    umbral_distancia = 0.1
    distancia_manos = calcular_distancia(landmark_izq, landmark_der)
    
    if distancia_manos < umbral_distancia:
        return True
    if distancia_manos > umbral_distancia:
        return False
def comprobar_makankosapo(hand_landmarks, umbral_distancia=0.001):
    # Verifica si la mano está frente a la cara (usando la posición de la nariz)
    muñeca = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    nariz = hand_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
    
    # Calcular la distancia entre la muñeca y la nariz
    distancia_muneca_nariz = calcular_distancia(muñeca, nariz)

    # Comprobar si la muñeca está a una distancia razonable de la nariz
    if distancia_muneca_nariz < umbral_distancia:
        # Verificar que los dedos índice y medio estén extendidos
        dedo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        nudillo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        dedo_medio = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        nudillo_medio = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Calcular distancias para los dedos índice y medio
        distancia_indice = calcular_distancia(dedo_indice, nudillo_indice)
        distancia_medio = calcular_distancia(dedo_medio, nudillo_medio)

        # Comprobar si los dedos índice y medio están extendidos
        if distancia_indice > 0.1 and distancia_medio > 0.1:  # Ambos dedos extendidos
            # Verificar que ningún otro dedo esté extendido
            dedos_no_permitidos = [
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            ]
            
            # Comprobar que los otros dedos no estén extendidos
            for dedo in dedos_no_permitidos:
                # Acceder al nudillo correspondiente
                if dedo == hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]:
                    nudillo = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                elif dedo == hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]:
                    nudillo = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                elif dedo == hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]:
                    nudillo = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                
                distancia_otro_dedo = calcular_distancia(dedo, nudillo)
                if distancia_otro_dedo > 0.1:  # Si otro dedo está extendido
                    return False
            
            # Si solo el índice y el medio están extendidos
            return True

    return False
# Función para comprobar la pose de Kikoho
def calcular_centroide(p1, p2, p3, p4):
    x = (p1.x + p2.x + p3.x + p4.x) / 4
    y = (p1.y + p2.y + p3.y + p4.y) / 4
    return x, y


def comprobar_mesenko(
    hand_landmarks_izquierda, hand_landmarks_derecha, 
    umbral_distancia=0.05, umbral_angulo=35, 
    umbral_cercania_indices=0.1, umbral_cercania_pulgares=0.1
):
    # Puntos de interés para el nuevo gesto Kikoho
    muñeca_izquierda = hand_landmarks_izquierda.landmark[mp_hands.HandLandmark.WRIST]
    muñeca_derecha = hand_landmarks_derecha.landmark[mp_hands.HandLandmark.WRIST]
    pulgar_izquierdo = hand_landmarks_izquierda.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pulgar_derecho = hand_landmarks_derecha.landmark[mp_hands.HandLandmark.THUMB_TIP]
    indice_izquierdo = hand_landmarks_izquierda.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    indice_derecho = hand_landmarks_derecha.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Calcular distancias clave
    distancia_pulgares = calcular_distancia(pulgar_izquierdo, pulgar_derecho)
    distancia_pulgar_izquierdo_indice_izquierdo = calcular_distancia(pulgar_izquierdo, indice_izquierdo)
    distancia_pulgar_derecho_indice_derecho = calcular_distancia(pulgar_derecho, indice_derecho)
    distancia_indices = calcular_distancia(indice_izquierdo, indice_derecho)
 
    # Condición 1: Los pulgares deben estar cerca uno del otro y los índices también
    # Además, los índices deben estar suficientemente separados de sus respectivos pulgares
    if (distancia_pulgares < umbral_cercania_pulgares and  # Pulgares deben estar cerca
        distancia_indices < umbral_cercania_indices and  # Índices deben estar cerca
        distancia_pulgar_izquierdo_indice_izquierdo > umbral_distancia and
        distancia_pulgar_derecho_indice_derecho > umbral_distancia):
        
        # Condición 2: Asegurar que los índices están por encima de los pulgares (más altos en el eje y)
        if indice_izquierdo.y < pulgar_izquierdo.y and indice_derecho.y < pulgar_derecho.y:
            
            # Calcular el ángulo entre los pulgares y los índices usando el método del coseno
            angulo_izquierdo = calcular_angulo(pulgar_izquierdo, indice_izquierdo, muñeca_izquierda)
            angulo_derecho = calcular_angulo(pulgar_derecho, indice_derecho, muñeca_derecha)
          
            # Condición 3: Restricción de los ángulos entre los pulgares e índices
            if angulo_izquierdo >= umbral_angulo and angulo_derecho >= umbral_angulo:
                # Si todas las condiciones son verdaderas, se detecta el Kikoho
                x_centro, y_centro = calcular_centroide(pulgar_izquierdo, pulgar_derecho, indice_izquierdo, indice_derecho)
                return True, x_centro, y_centro

    return False, None, None


# LO QUE DICE EL NOMBRE
#          |
#          |
#          |
#          |
#          V
def superponer_imagen(fondo, imagen, x, y):
    h_fondo, w_fondo, _ = fondo.shape
    h_img, w_img = imagen.shape[:2]

    # VEO QUE NO SE SALGA DEL CUADRO PARA QUE NO CRASHEE A LA MIERDA 
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w_img > w_fondo:
        w_img = w_fondo - x
    if y + h_img > h_fondo:
        h_img = h_fondo - y

    # CORTO LA IMAGEN SI SE SALE
    imagen = imagen[:h_img, :w_img]

    # CALCULAR TRANSPARENCIA
    if imagen.shape[2] == 4:  # PNG CON TRANSPARENCIA
        alpha_s = imagen[:, :, 3] / 255.0  # CANAL ALFA (GIGACHAD, SKIBIDI, SIGMA, POMNI)
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):  # RGB (ESO LO HACE GAMER)
            fondo[y:y+h_img, x:x+w_img, c] = (alpha_s * imagen[:, :, c] +
                                              alpha_l * fondo[y:y+h_img, x:x+w_img, c])
    else:
        fondo[y:y+h_img, x:x+w_img] = imagen

# INICIALIZO LAS VARIABLES DEL SONIDO
sonido_reproduciendo = False
sonido_ki_reproduciendo = False
sonido_kamehameha_reproduciendo = False
kamehameha_activado = False
tiempo_inicio = 0
tiempo_transcurrido = 0
flash_activado = False
tiempo_flash_inicio = 0
masenko_activado = False

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error al capturar el frame")
            break
        
        # CONVERTIR A RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # PROCESAR FRAMES
        pose_results = pose.process(rgb_frame)

        # PROCESAR FRAMES PARA MASNO
        hands_results = hands.process(rgb_frame)

        # VARIABLE PARA LOS MENSAJINHOS
        mensaje = ""

        # DEFINO LOS ESTADOS DE LA VAINA
        # (HACEN LO QUE DICE LOS NOMBRES NO ROMPAS LA BOLAS)
        brazo_derecho_doblado = False
        brazo_izquierdo_doblado = False
        puño_derecho_cerrado = False
        puño_izquierdo_cerrado = False
        kamehameha_detectado = False 
        sonido_actual = None  
        sonido_makankosapo_reproduciendo = False 
        



        # LOS PUNTINHOS
        if pose_results.pose_landmarks:
            #mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # HOMBROS CODOS Y MUÑECA
            landmarks = pose_results.pose_landmarks.landmark
            hombro_derecho = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            codo_derecho = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            muñeca_derecha = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            hombro_izquierdo = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            codo_izquierdo = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            muñeca_izquierda = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

            # NAZO
            nariz = landmarks[mp_pose.PoseLandmark.NOSE.value]

            # ANGULO DE CODO
            angulo_derecho = calcular_angulo(hombro_derecho, codo_derecho, muñeca_derecha)
            angulo_izquierdo = calcular_angulo(hombro_izquierdo, codo_izquierdo, muñeca_izquierda)

            # ESTA DOBLAO
            if angulo_derecho < 160:
                mensaje += ""
                brazo_derecho_doblado = True
            if angulo_izquierdo < 160:
                mensaje += ""
                brazo_izquierdo_doblado = True

        # PUNTINHOS DE MANOS
       # PUNTINHOS DE MANOS
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    dedo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    h, w, _ = frame.shape
                    x_indice = int(dedo_indice.x * w)  # Asegúrate de usar el ancho del frame
                    y_indice = int(dedo_indice.y * h)  # Asegúrate de usar la altura del frame

                    # Superponer la imagen en la punta del dedo índice
                    if comprobar_makankosapo(hand_landmarks) and not masenko_activado and not kamehameha_activado:
                        mensaje += ""
                        makankosapo_detectado = True  # Se detectó el Makankosapo
                        if not sonido_makankosapo_reproduciendo:
                            makankosapo_sound.play()
                            sonido_makankosapo_reproduciendo = True  # Marca que el sonido se está reproduciendo

                        # Mostrar el mensaje en la pantalla
                        cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        # Superponer la imagen del Makankosapo si se detectó
                        if makankosapo_detectado:
                            imagen_dedo_indice = cv2.imread('makankosapo.png', cv2.IMREAD_UNCHANGED)  # Asegúrate de que la imagen tenga un canal alfa
                            h_img, w_img = imagen_dedo_indice.shape[:2]
                            # Centrar la imagen en la punta del dedo índice
                            superponer_imagen(frame, imagen_dedo_indice, x_indice - w_img // 2, y_indice - h_img // 2)
                                
                # PUÑO CERRADO (MENSAJE)
                if comprobar_puño_cerrado(hand_landmarks):
                    mensaje += ""
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x:
                        puño_izquierdo_cerrado = True
                    else:
                        puño_derecho_cerrado = True
        if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) >= 2:
    # Obtener las dos primeras manos detectadas (izquierda y derecha)
            hand_landmarks_izquierda = hands_results.multi_hand_landmarks[0]
            hand_landmarks_derecha = hands_results.multi_hand_landmarks[1]

            # Dibujar las conexiones de cada mano en el frame
            #mp_drawing.draw_landmarks(frame, hand_landmarks_izquierda, mp_hands.HAND_CONNECTIONS)
            #mp_drawing.draw_landmarks(frame, hand_landmarks_derecha, mp_hands.HAND_CONNECTIONS)
                            
            # Verificar si las manos forman el Kikoho y obtener la posición central del triángulo
            kikoho_detectado, x_centro, y_centro = comprobar_mesenko(hand_landmarks_izquierda, hand_landmarks_derecha)

            if kikoho_detectado and not kamehameha_activado:
                mensaje += ""
                masenko_activado = True          
                masenko_sound.play()
                # Convertir las coordenadas del centro del triángulo a píxeles
                h, w, _ = frame.shape
                x_centro_px = int(x_centro * w)
                y_centro_px = int(y_centro * h)

                # Redimensionar y superponer la imagen del Kikoho
                kikoho_image_resized = cv2.resize(masenko_image, (200, 200))  # Ajusta el tamaño si es necesario
                superponer_imagen(frame, kikoho_image_resized, x_centro_px - 100, y_centro_px - 100)  # Centra la imagen en el triángulo
            else:
                masenko_activado = False
                masenko_sound.stop()


        # FRAMES MANOS (CREO QUE ESTA REPETIDO PERO NO LO BORRO POR QUE TENGO MIEDO DE QUE EXPLOTE)
        hands_results = hands.process(rgb_frame)

        if hands_results.multi_hand_landmarks:
            mano_izquierda_detectada = False
            mano_derecha_detectada = False
            for hand_landmarks in hands_results.multi_hand_landmarks:
               # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # MANO ABIERTA MENSAJE (LE PUSE "Q" POR QUE ME PINTO)
                if not comprobar_puño_cerrado(hand_landmarks):
                    mensaje += ""
                    if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) == 2:
                        mano_izquierda = hands_results.multi_hand_landmarks[0]
                        mano_derecha = hands_results.multi_hand_landmarks[1]

    
               
                            # MANO JUNTA LLAMAR
                        if manos_juntas(mano_izquierda.landmark[mp_hands.HandLandmark.WRIST], mano_derecha.landmark[mp_hands.HandLandmark.WRIST]):
                            kamehameha_detectado = True
                            mensaje += ""
                            
                            if kamehameha_detectado and not flash_activado:
                                # Activa el sonido y visualización del Kamehameha si no están activos
                                if not sonido_kamehameha_reproduciendo:
                                    kamehameha_sound.play()
                                    sonido_kamehameha_reproduciendo = True

                                # Comienza el tiempo de activación del Kamehameha si no está en curso
                                if not kamehameha_activado:
                                    tiempo_inicio = time.time()
                                    kamehameha_activado = True

                                # Calcula el tiempo transcurrido para activar el flash
                                tiempo_transcurrido = time.time() - tiempo_inicio
                                if tiempo_transcurrido >= 4:
                                    flash_activado = True
                                    tiempo_flash_inicio = time.time()  # Tiempo de inicio del flash

                            # Si el flash está activo, apaga el Kamehameha y su sonido
                            if flash_activado:
                                # Detiene el sonido del Kamehameha y resetea las variables
                                if sonido_kamehameha_reproduciendo:
                                    kamehameha_sound.stop()
                                    sonido_kamehameha_reproduciendo = False
                                    ex_sound.play()
                                kamehameha_activado = False  # Desactiva el Kamehameha visualmente

                                # Duración del flash (puedes ajustar la duración en segundos)
                                tiempo_flash_transcurrido = time.time() - tiempo_flash_inicio
                                if tiempo_flash_transcurrido < 1:
                                    frame[:, :] = 255  # Efecto de flash blanco inicial
                                elif tiempo_flash_transcurrido < 3:
                            # Degradado de blanco a nada (desvanecimiento de flash)
                                    alpha = (tiempo_flash_transcurrido - 1) / 2  # Alpha aumenta de 0 a 1
                                    frame = cv2.addWeighted(255 * np.ones_like(frame, dtype=np.uint8), 1 - alpha, frame, alpha, 0)
                                else:
                                    flash_activado = False 

                            # Visualización del Kamehameha si el flash no está activo
                            if kamehameha_activado and not flash_activado:
                                    # LA IMAGEN SE MUEVE CON LA MUÑECA (NO VA A SER CON LA CHOT*, NO?)
                                h, w, _ = frame.shape
                                muñeca_x = int(muñeca_derecha.x * w)  # X MUÑECA (DERECHA PORQUE YO NUNCA VOY A LA IZQUIERDA VLLC)
                                muñeca_y = int(muñeca_derecha.y * h)  # LO MUSMO PERO Y
                                if tiempo_transcurrido < 7:
                                    # Ajustar y centrar la imagen
                                    Tamanox = int(450 + (tiempo_transcurrido ** 1.5) * 10)
                                    Tamanoy = int(600 + (tiempo_transcurrido ** 1.5) * 10)
                                else:
                                    Tamanox = int(450 + (7 ** 1.5) * 10)
                                    Tamanoy = int(600 + (7** 1.5) * 10)

                                kamehameha_image_resized = cv2.resize(kamehameha_image, (Tamanox, Tamanoy)) 

                                        # CENTRAR IMAGEN (COMO LOS CENTROS DE ADVINCULA)
                                    
                                x_offset = muñeca_x - kamehameha_image_resized.shape[1] // 2  # CENTRO AL X
                                y_offset = muñeca_y - kamehameha_image_resized.shape[0] // 2  # CENTRO AL Y

                # PONER LA IMAGEN
                                superponer_imagen(frame, kamehameha_image_resized, x_offset, y_offset)

                            else:
                                # PARALE CUMBIA  
                                if sonido_kamehameha_reproduciendo:
                                    kamehameha_sound.stop()
                                    sonido_kamehameha_reproduciendo = False
                                    kamehameha_activado = False 

        # COMPROBAR SI ESTA CAGANDO KI... NO!... EHH... CARGANDO KI... ESO ERA.. JEJE
             
        if not kamehameha_detectado and brazo_derecho_doblado and brazo_izquierdo_doblado and puño_derecho_cerrado and puño_izquierdo_cerrado:
            mensaje += ""

    # PONELE CUMBIA PERO CARGANDO KI
            if not sonido_ki_reproduciendo:
                ki_sound.play()
                sonido_ki_reproduciendo = True
            if sonido_kamehameha_reproduciendo:
                kamehameha_sound.stop()
                sonido_kamehameha_reproduciendo = False
    # PONER LA IMAGEN SEGUN EL NAZO
            h, w, _ = frame.shape
            nariz_x = int(nariz.x * w)
            nariz_y = int(nariz.y * h)

            ki_image_resized = cv2.resize(ki_image, (450, 600))  # AJUSTA
            x_offset = nariz_x - ki_image_resized.shape[1] // 2  # CENTRO X
            y_offset = nariz_y - ki_image_resized.shape[0] // 2  # CENTRO Y

    # Superponer la imagen de KI en la posición calculada
            superponer_imagen(frame, ki_image_resized, x_offset, y_offset)


        else:
            # PARALE CUMBIA PERO CARGANDO KI
            if sonido_ki_reproduciendo:
                ki_sound.stop()
                sonido_ki_reproduciendo = False
        
        if kamehameha_detectado:
            tiempo_transcurrido = time.time() - tiempo_inicio
            mensaje += ""


        # PONER EL MENSAJINHO
        cv2.putText(frame, mensaje, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # PONER EL FRAME
        cv2.imshow('Detección de movimientos', frame)


        # SALIR CON LA Q (NO CON LA "X" PORQUE MESSIENTOTROLL)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # TENES QUE CERRAR EL ESTADIO... LO GENIO... LO GENIO HACEN ESO
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit() 
