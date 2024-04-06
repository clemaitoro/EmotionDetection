import cv2
import numpy as np
from keras.models import load_model

# Load your trained emotion recognition model
emotion_model = load_model("/home/user/persistent/final_stretch/model.h5")

def preprocess_face(face_image):
    # Resize, convert to 3 channels if grayscale, normalize and expand dimensions
    face_resized = cv2.resize(face_image, (48, 48))
    if len(face_resized.shape) == 2 or face_resized.shape[2] == 1:
        face_resized = np.stack((face_resized,) * 3, axis=-1)
    face_normalized = face_resized / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=0)
    return face_expanded

def predict_emotion(face_image, model):
    preprocessed_face = preprocess_face(face_image)
    prediction = model.predict(preprocessed_face)
    return np.argmax(prediction)

def read_yolo_bboxes(file_path, image_width, image_height):
    yolo_bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            _, center_x, center_y, width, height = map(float, line.split())
            x_min = int((center_x - width / 2) * image_width)
            y_min = int((center_y - height / 2) * image_height)
            x_max = int((center_x + width / 2) * image_width)
            y_max = int((center_y + height / 2) * image_height)
            yolo_bboxes.append([x_min, y_min, x_max, y_max])
    return yolo_bboxes

# Load the image
image_path = "/home/user/runs/detect/exp2/57680ecc003eaaa0.jpg"
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Read bounding boxes from a YOLO format file
bbox_file_path = "/home/user/runs/detect/exp2/labels/57680ecc003eaaa0.txt"
yolo_bounding_boxes = read_yolo_bboxes(bbox_file_path, image_width, image_height)

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

for bbox in yolo_bounding_boxes:
    x_min, y_min, x_max, y_max = bbox
    face_image = image[y_min:y_max, x_min:x_max]
    emotion_index = predict_emotion(face_image, emotion_model)
    emotion_label = emotions[emotion_index]

    # Annotate the image with the bounding box and emotion label
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, emotion_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

# Display the annotated image
cv2.imshow("Emotion Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
