import cv2
import os
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
    CASC_PATH = 'haarcascade_frontalface_default.xml'
    FACE_CASCADE = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), CASC_PATH))

    def get_images_with_faces(self, config):
        images_faces = []
        for row in config['images']:
            image = self.read_image(row['path'])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.detect_multi_scale(gray, scale_factor=row.get('scale_factor'),
                                            min_neighbors=config['min_neighbors'], min_size=config['min_size'])
            images_faces.append((image, faces))
        return images_faces

    @staticmethod
    def read_image(image_path):
        aa = os.path.join(os.path.dirname(__file__), image_path)
        return cv2.imread(aa)

    def detect_multi_scale(self, gray, scale_factor, min_neighbors, min_size):
        return self.FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    @staticmethod
    def print_single_image(image, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('', image)
        cv2.waitKey(0)

    def print_images(self, images_faces):
        for image, faces in images_faces:
            self.print_single_image(image, faces)


# face_recognition = FaceRecognition()
# config = face_recognition.read_config('config.yaml')
# images_with_faces = face_recognition.get_images_with_faces(config)
# face_recognition.print_images(images_with_faces)
