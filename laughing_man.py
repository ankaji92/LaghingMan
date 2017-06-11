#!/usr/local/bin/python3
#coding: utf-8

import cv2
import numpy as np

def make_mask(laugh):
    mask = np.zeros(laugh.shape)
    for i, line in enumerate(laugh):
        non_white_pxs = np.array(np.where(line < 250))
        if non_white_pxs.size > 0:
            min_h, max_h = np.min(non_white_pxs, axis=1)[0], np.max(non_white_pxs, axis=1)[0]
            mask[i, min_h:max_h, :] = 1

    return mask

cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

gif_cap = cv2.VideoCapture("laughing_man.gif")
_, laugh = gif_cap.read()
mask = make_mask(laugh)

while ret:

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 画像認識を高速に行うためにグレースケール化。
    gray = cv2.resize(gray, (int(frame.shape[1]/4), int(frame.shape[0]/4)))

    faces = cascade.detectMultiScale(gray)  # 顔を探す。

    if len(faces) > 0:
        for rect in faces:
            rect *= 4

            laugh = cv2.resize(laugh, tuple(rect[2:]))
            mask = cv2.resize(mask, tuple(rect[2:]))

            # 笑い男の合成。
            frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :] = laugh[:,:, :] * mask + frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :] * (1.0 - mask)

    cv2.imshow('laughing man', frame)

    ret, frame = cap.read()

    gif_cap.grab()
    gif_cap.grab()
    gif_ret, laugh = gif_cap.read()
    if not gif_ret:
        gif_cap = cv2.VideoCapture("laughing_man.gif")
        gif_ret, laugh = gif_cap.read()

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
