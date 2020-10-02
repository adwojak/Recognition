import cv2

# image_path = './assets/abba.png'
# image_path = './assets/important.jpg'
image_path = './assets/smiling_people.jpg'
CASC_PATH = './haarcascade_frontalface_default.xml'
FACE_CASCADE = cv2.CascadeClassifier(CASC_PATH)


def get_image(image_path):
    return cv2.imread(image_path)


def get_faces_from_image_path(image_path):
    image = get_image(image_path)
    return get_faces_from_image(image)


def get_faces_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.13,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )


def print_faces(image_path):
    image = get_image(image_path)
    faces = get_faces_from_image(image)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('', image)
    cv2.waitKey(0)


print_faces(image_path)
