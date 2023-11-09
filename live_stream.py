import cv2
import time

from insightface_face_detection import IFRClient
from yolov8_mobile_detection import mobile_phone_detection

# Create an instance of the IFRClient class
client = IFRClient()

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://grilsquad:grilsquad@192.168.18.93:554/stream1")
# cap.set(3, 640)
# cap.set(3, 360)
# cap = cv2.VideoCapture("http://192.168.15.165:4747/video")

batch_of_frames = []

# Check if camera opened successfully
if cap.isOpened() is False:
    print("Unable to read camera feed")
else:
    new_frame_time = 0
    previous_frame_time = 0
    while True:
        new_frame_time = time.time()
        ret, frame = cap.read()

        # Resize the frame
        frame = cv2.resize(frame, (640, 360))

        # while len(batch_of_frames) < 8:
        #     batch_of_frames.append(frame)

        # Detect mobile phone
        mobile_phone_locations = mobile_phone_detection(frame)
        # mobile_phone_locations = batch_mobile_phone_detection(batch_of_frames)
        # print(f"Mobile Phone Locations: {mobile_phone_locations}")
        # print()
        # Detect faces
        face_locations = client.face_locations(frame)
        # face_locations = client.batch_face_locations(batch_of_frames)
        # print(f"Face Locations: {face_locations}")
        # print()
        # Loop through detected faces
        for face in face_locations:
            # Get coordinates of the face
            x_min, y_min, x_max, y_max = face

            # Check if the face region is entirely contained within the mobile region
            # print(mobile_phone_locations)
            if mobile_phone_locations:
                face_within_cellphone = any(
                    x_min >= x1 and y_min >= y1 and x_max <= x2 and y_max <= y2 for x1, y1, x2, y2 in mobile_phone_locations
                )

                if face_within_cellphone:
                    label = 'spoof'
                    # print('spoof')
                    # Display the label on the frame
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # Draw a box around each face and display it
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                else:
                    label = 'real'
                    # print('real')
                    # Display the label on the frame
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # Draw a box around each face and display it
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            else:
                label = 'real'
                # print('real')
                # Display the label on the frame
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Draw a box around each face and display it
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow('frame', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        fps = 1 / (new_frame_time - previous_frame_time)
        print(f"FPS: {fps}")
        previous_frame_time = new_frame_time

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
