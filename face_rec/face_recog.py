import cv2
import face_recognition
import os
import pickle

face_data_dir = 'face_data'
if not os.path.exists(face_data_dir):
    os.makedirs(face_data_dir)

known_face_encodings = []
known_face_names = []


for filename in os.listdir(face_data_dir):
    if filename.endswith('.pkl'):
        with open(os.path.join(face_data_dir, filename), 'rb') as f:
            data = pickle.load(f)
            known_face_encodings.append(data['encoding'])
            known_face_names.append(data['name'])

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        
        face_names.append(name)
        
        if name == "Unknown":
            cv2.imshow('Video', frame)
            name = input("Enter the name of the new face: ")
            
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            data = {'encoding': face_encoding, 'name': name}
            with open(os.path.join(face_data_dir, f'{name}.pkl'), 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved encoding for {name}")
            
            # Save the face image
            face_image = frame[top*4:bottom*4, left*4:right*4]
            face_image_filename = os.path.join(face_data_dir, f'{name}.jpg')
            cv2.imwrite(face_image_filename, face_image)
            print(f"Saved image for {name}")

    # Draw bounding boxes and names on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
