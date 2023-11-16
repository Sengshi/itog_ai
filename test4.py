import cv2
import face_recognition
import numpy as np

# Загрузка известных изображений
known_faces = []
known_names = []


def access_office():
    pass


for i in range(100):
    image = face_recognition.load_image_file(f"database/person ({i + 1}).jpg")
    try:
        # Изображения могут не содержать лиц, поэтому добавляем обработку исключений
        face_encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(face_encoding)
        known_names.append(f"Person ({i + 1})")
    except IndexError:
        print(f"No faces found in the image at path: database/person ({i + 1}).jpg")


# Сравнение лиц
def check_spoof(face_encoding):
    if len(known_faces) > 0:
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        min_distance_index = int(face_distances.argmin())
        if face_distances[min_distance_index] < 0.6:
            return known_names[min_distance_index]
    return "Unknown"


frame = cv2.imread('testing/test_image.jpg')  # Получение кадра


# Используем библиотеку face_recognition для определения лиц на изображении.
rgb_frame = frame[:, :, ::-1]  # cv2 использует BGR, face_recognition - RGB
face_locations = face_recognition.face_locations(rgb_frame)
face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    name = check_spoof(face_encoding)
    if name != "Unknown":
        print('Доступ разрешен')
        access_office()
    else:
        print('Доступ запрещен')

    # Рисуем рамку вокруг лица и выводим имя
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)

# Выводим изображение в окне
cv2.imshow('Face Recognition', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
