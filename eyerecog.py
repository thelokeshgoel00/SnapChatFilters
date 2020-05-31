import numpy as np
import cv2
from utility import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("Train/third-party/haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("Train/third-party/frontalEyes35x16.xml")
nose_cascade = cv2.CascadeClassifier("Train/third-party/Nose18x15.xml")
glasses = cv2.imread("Train/glasses.png", -1)
mustache = cv2.imread("Train/mustache.png", -1)
skip = 0
fps = 10
save_path = 'media/watermark.mp4'
#config = CFEVideoConf(cap, filepath=save_path, res='480p')
#out = cv2.VideoWriter(save_path, config.video_type, fps, config.dims)
face_data = []
dataset_path = 'MyData/'
print('I')
#file_name = input("Enter the Person name")
while True:
    # frame = cv2.imread("Test/Before.png")
    ret, frame = cap.read()
    if ret == False:
        print("F")
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #faces = sorted(faces, key=lambda f: f[2] * f[3])
    face_section = []

    frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2BGRA)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        eyes = eyes_cascade.detectMultiScale(frame, 1.3,5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex + ew, y+ey + eh), (0, 255, 0), 2)
            roi_eyes = roi_color[ey:ey+eh, ex:ex+ew]
            glasses = image_resize(glasses.copy() , width=ew)

            gw, gh, gc = glasses.shape
            for i in range(0,gw):
                for j in range(0,gh):
                    if glasses[i,j][3]!=0:
                        frame[ey+i,ex+j]=glasses[i,j]

        nose = nose_cascade.detectMultiScale(roi_gray,1.3,5)
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(frame, (x+nx, y+ny), (x+nx + nw, y + ny + nh), (0, 0, 255), 2)
            roi_eyes = roi_gray[ny:ny+nh, nx:nx+nw]
            mustache2 = image_resize(mustache.copy(), width = int(1.2*nw))
            mw, mh,mc = mustache2.shape

            for i in range(0,mw):
                for j in range(0,mh):
                    if mustache2[i,j][3]!=0:
                        roi_color[ny+i+ int(nh/2.0),nx+j] = mustache2[i,j]
    cv2.imshow("rftt", frame)
    cv2.waitKey(0)
    #     offset = 10
    #     frame = np.array(frame)
    #
    #     face_section = frame[y - offset:y + h + offset, x - offset:x + w + offset]
    #     face_section = cv2.resize(face_section, (100, 100))
    #     skip += 1
    #     if skip % 10 == 0:
    #         face_data.append(face_section)
    #         print(len(face_data))
    # cv2.imshow("Video Frame", frame)
    # if len(face_section) >= 1:
    #     cv2.imshow("Face Section", face_section)

    key_pressed = cv2.waitKey(0) & 0xFF
    if key_pressed == ord('q'):
        break


#np.save(dataset_path + file_name + '.npy', face_data)
print("Data saved successfully")
#cap.release()
cv2.destroyAllWindows()
