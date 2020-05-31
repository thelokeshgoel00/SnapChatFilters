import numpy as np
import pandas as pd
import cv2

from utility import image_resize, CFEVideoConf

cap = cv2.VideoCapture(0)
img_path = 'encoder.jpg'
logo = cv2.imread(img_path, -1)
logo = np.array(logo)
print(logo.shape)

fps = 10
save_path='media/watermark.mp4'
config = CFEVideoConf(cap, filepath=save_path, res = '480p')
out = cv2.VideoWriter(save_path, config.video_type, fps, config.dims)
watermark = image_resize(logo, height=50)
print(watermark.shape)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)

cv2.imshow("watermark", watermark)
print(watermark.shape)
while (True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    frame_h, frame_w, frame_c = frame.shape
    print(frame.shape)
    overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    print(overlay.shape)
    #overlay[100:250, 100:250] = (255, 255, 0, 10)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # overlay = np.zeroes()
    # cv2.imshow("overlay",overlay)
    watermark_h, watermark_w, watermark_c = watermark.shape

    for i in range(frame_h-watermark_h, frame_h):
        for j in range(frame_w-watermark_w, frame_w):
            #print(watermark[i,j])
            if watermark[i-frame_h+watermark_h, j-frame_w+watermark_w][3]!=0:
                overlay[i, j] = watermark[i-frame_h+watermark_h, j-frame_w+watermark_w]
    cv2.imshow("overlay", overlay)
    cv2.addWeighted(overlay, 0.25, frame, 1.0, 0, frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    out.write(frame)
    cv2.imshow("My window", frame)
cap.release()
out.release()
cv2.destroyAllWindows()
