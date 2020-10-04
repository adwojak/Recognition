import os
from face_recognition.main import FaceRecognition


face_recognition = FaceRecognition()
config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.yaml')
config = face_recognition.read_config(config_path)
images = face_recognition.get_images_with_faces(config)

for index, image_with_faces in enumerate(images):
    assert len(image_with_faces[1]) == config['images'][index]['faces_count']
