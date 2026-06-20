import cv2
import pandas as pd
from datetime import datetime
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

attendance_marked = False

while True:

    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            "Saakshi",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )

        if not attendance_marked:

            now = datetime.now()

            time = now.strftime("%H:%M:%S")

            if not os.path.exists("attendance.csv"):

                df = pd.DataFrame(
                    [["Saakshi", time]],
                    columns=["Name", "Time"]
                )

                df.to_csv(
                    "attendance.csv",
                    index=False
                )

            else:

                df = pd.read_csv(
                    "attendance.csv"
                )

                new_row = pd.DataFrame(
                    [["Saakshi", time]],
                    columns=["Name", "Time"]
                )

                df = pd.concat(
                    [df, new_row],
                    ignore_index=True
                )

                df.to_csv(
                    "attendance.csv",
                    index=False
                )

            attendance_marked = True

    cv2.imshow(
        "AI Attendance Tracker",
        img
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()