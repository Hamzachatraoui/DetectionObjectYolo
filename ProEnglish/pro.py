# import cv2
# import numpy as np

# # load our YOLO object detector trained on COCO dataset (80 classes)
# net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


# DEFAULT_CONFIANCE = 0.5
# THRESHOLD = 0.4
# # load the COCO class labels our YOLO model was trained on
# with open('coco.names', 'r') as f:
#     LABELS = f.read().splitlines()

# # initialize the video stream, pointer to output video file
# cap = cv2.VideoCapture(0)

# while True:
#     _,image=cap.read()
#     height, width, _ = image.shape

#     blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
#     net.setInput(blob)
#     layerOutputs = net.forward(ln)

#     # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
#     boxes = []
#     confidences = []
#     classIDs = []

#     # loop over each of the layer outputs
#     for output in layerOutputs:
#         # loop over each of the detections
#         for detection in output:
#             # extract the class ID and confidence (i.e., probability)
#             # of the current object detection
#             scores = detection[5:]
#             classID = np.argmax(scores)
#             confidence = scores[classID]
#             # filter out weak predictions by ensuring the detected
#             # probability is greater than the minimum probability
#             if confidence > DEFAULT_CONFIANCE:
#                 # scale the bounding box coordinates back relative to
#                 # the size of the image, keeping in mind that YOLO
#                 # actually returns the center (x, y)-coordinates of
#                 # the bounding box followed by the boxes' width and
#                 # height
#                 box = detection[0:4] * np.array([width, height, width, height])
#                 (centerX, centerY, W, H) = box.astype("int")
#                 # use the center (x, y)-coordinates to derive the top
#                 # and and left corner of the bounding box
#                 x = int(centerX - (W / 2))
#                 y = int(centerY - (H / 2))
#                 # update our list of bounding box coordinates,
#                 # confidences, and class IDs
#                 boxes.append([x, y, int(W), int(H)])
#                 confidences.append(float(confidence))
#                 classIDs.append(classID)
        
#     # apply non-maxima suppression to suppress weak, overlapping
#     # bounding boxes
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)

#     # initialize a list of colors to represent each possible class label
#     COLORS = np.random.uniform(0,255,size=(len(boxes), 3))

#     # ensure at least one detection exists
#     if len(indexes) > 0:
#         # loop over the indexes we are keeping
#         for i in indexes.flatten():
#             # extract the bounding box coordinates
#             (x, y, w, h) = boxes[i]
#             # draw a bounding box rectangle and label on the frame
#             color = COLORS[i]
#             text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
#             cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(image, text, (x, y + 20 ), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

#     cv2.imshow('Image', image)
#     if cv2.waitKey(1)==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import streamlit as st

# Charger le détecteur YOLO formé sur le jeu de données COCO
net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

DEFAULT_CONFIANCE = 0.5
THRESHOLD = 0.4

# Charger les étiquettes COCO
with open('coco.names', 'r') as f:
    LABELS = f.read().splitlines()

# Interface Streamlit
st.title("Détection d'objets en temps réel ")
st.text("Appuyez sur le bouton ci-dessous pour démarrer la caméra.")
start_button = st.button("Démarrer la caméra")

if start_button:
    # Initialiser le flux vidéo
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()  # Conteneur pour afficher les images

    while True:
        ret, image = cap.read()
        if not ret:
            st.warning("Impossible de lire la vidéo. Vérifiez votre caméra.")
            break

        height, width, _ = image.shape

        # Prétraitement de l'image pour YOLO
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        # Parcourir les sorties des couches
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > DEFAULT_CONFIANCE:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, W, H) = box.astype("int")
                    x = int(centerX - (W / 2))
                    y = int(centerY - (H / 2))
                    boxes.append([x, y, int(W), int(H)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # Appliquer la suppression non maximale
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)
        COLORS = np.random.uniform(0, 255, size=(len(boxes), 3))

        # Dessiner les boîtes détectées
        if len(indexes) > 0:
            for i in indexes.flatten():
                (x, y, w, h) = boxes[i]
                color = COLORS[i]
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convertir l'image en format RGB pour Streamlit
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st_frame.image(image_rgb, channels="RGB")

        # Arrêter si l'utilisateur ferme la caméra
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
