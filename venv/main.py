import cv2
import mediapipe as mp
import math
import pygame

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
# AGARRAR SONIDOS
ki_sound = pygame.mixer.Sound('ki_sound.wav') 
kamehameha_sound = pygame.mixer.Sound('kamehameha_sound.wav')


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
        



        # LOS PUNTINHOS
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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
                mensaje += "BD "
                brazo_derecho_doblado = True
            if angulo_izquierdo < 160:
                mensaje += "BI "
                brazo_izquierdo_doblado = True

        # PUNTINHOS DE MANOS
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # PUÑO CERRADO (MENSAJE)
                if comprobar_puño_cerrado(hand_landmarks):
                    mensaje += "P "
                    if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x:
                        puño_izquierdo_cerrado = True
                    else:
                        puño_derecho_cerrado = True

        # FRAMES MANOS (CREO QUE ESTA REPETIDO PERO NO LO BORRO POR QUE TENGO MIEDO DE QUE EXPLOTE)
        hands_results = hands.process(rgb_frame)

        if hands_results.multi_hand_landmarks:
            mano_izquierda_detectada = False
            mano_derecha_detectada = False
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # MANO ABIERTA MENSAJE (LE PUSE "Q" POR QUE ME PINTO)
                if not comprobar_puño_cerrado(hand_landmarks):
                    mensaje += "Q "
                    if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) == 2:
                        mano_izquierda = hands_results.multi_hand_landmarks[0]
                        mano_derecha = hands_results.multi_hand_landmarks[1]

    
               
                            # MANO JUNTA LLAMAR
                        if manos_juntas(mano_izquierda.landmark[mp_hands.HandLandmark.WRIST], mano_derecha.landmark[mp_hands.HandLandmark.WRIST]):
                            kamehameha_detectado = True
                            mensaje += "KAMEHAMEHA"
                                # PONELE CUMBIA (KAMEHAMEHA)
                            if not sonido_kamehameha_reproduciendo:
                                kamehameha_sound.play()
                                sonido_kamehameha_reproduciendo = True


                                # LA IMAGEN SE MUEVE CON LA MUÑECA (NO VA A SER CON LA CHOT*, NO?)
                            h, w, _ = frame.shape
                            muñeca_x = int(muñeca_derecha.x * w)  # X MUÑECA (DERECHA PORQUE YO NUNCA VOY A LA IZQUIERDA VLLC)
                            muñeca_y = int(muñeca_derecha.y * h)  # LO MUSMO PERO Y

                                    # AJUSTAR IMAGEN (PERO EN ENGLISH)
                            kamehameha_image_resized = cv2.resize(kamehameha_image, (450, 600))  

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

        # COMPROBAR SI ESTA CAGANDO KI... NO!... EHH... CARGANDO KI... ESO ERA.. JEJE
             
        if not kamehameha_detectado and brazo_derecho_doblado and brazo_izquierdo_doblado and puño_derecho_cerrado and puño_izquierdo_cerrado:
            mensaje += "CARGANDO KI"

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
