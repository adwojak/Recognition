import cv2
from yaml import safe_load


class Base:
    def read_config(self, file: str) -> dict:
        with open(file, 'r') as yaml_config:
            config_raw = safe_load(yaml_config)
        config_raw['min_size'] = (config_raw['min_width'], config_raw['min_height'])
        del config_raw['min_width']
        del config_raw['min_height']
        return config_raw


class FaceRecognition(Base):
    CASC_PATH = './haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(CASC_PATH)

    def get_images_with_faces(self, config):
        images_with_faces = []
        for row in config['images']:
            image = self.read_image(row['path'])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detect_multi_scale(gray, scale_factor=row.get('scale_factor'),
                                            min_neighbors=config['min_neighbors'], min_size=config['min_size'])
            images_with_faces.append((image, faces))
        return images_with_faces

    def read_image(self, image_path):
        return cv2.imread(image_path)

    def detect_multi_scale(self, gray, scale_factor, min_neighbors, min_size):
        return self.FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    def print_face(self, image, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('', image)
        cv2.waitKey(0)


face_recognition = FaceRecognition()
config = face_recognition.read_config('config.yaml')
images_with_faces = face_recognition.get_images_with_faces(config)
face_recognition.print_face(*images_with_faces[0])
