import cv2
import face_recognition
import os


def access_office():
    pass


# Сравнение лиц
def check_spoof(face_encoding):
    if len(known_faces) > 0:
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        min_distance_index = int(face_distances.argmin())
        if face_distances[min_distance_index] < 0.5:
            return known_names[min_distance_index]
    return "Unknown"


if __name__ == '__main__':
    known_faces = []
    known_names = []

    # Загрузка известных изображений
    db = 'database'
    for filename in os.listdir(db):
        f = os.path.join(db, filename)
        if os.path.isfile(f):
            image = face_recognition.load_image_file(f)
            # Изображения могут не содержать лиц, поэтому добавляем обработку исключений
            try:
                face_encoding = face_recognition.face_encodings(image)[0]
                known_faces.append(face_encoding)
                known_names.append(f'{filename.title().rsplit(".", 1)[0]}')
            except IndexError:
                print(f"Лица не обнаружены в файле по пути: {f}")

    # Открытие потока видео с камеры
    cap = cv2.VideoCapture(0)
    frame = cap  # Получение кадра с камеры

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Используем библиотеку face_recognition для определения лиц на изображении.
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
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1)

        # Выводим изображение в окне
        cv2.imshow('Face Recognition', frame)

        # Для завершения работы программы нажмите 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()
