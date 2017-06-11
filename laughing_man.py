#!/usr/local/bin/python3
#coding: utf-8

import cv2
import numpy as np


# Segment Laughing Man (to make gif transparent)
def segment_lm(lm):
    lm_region = np.zeros(lm.shape)
    for v, line in enumerate(lm):
        non_white_pxs = np.array(np.where(line < 250))
        if non_white_pxs.size > 0:
            min_h, max_h = np.min(non_white_pxs, axis=1)[0], np.max(non_white_pxs, axis=1)[0]
            lm_region[v, min_h:max_h, :] = 1

    segmented_lm = lm_region * lm

    return segmented_lm, lm_region


# Synthesize Laughing Man frame
def synth_lm(frame, segmented_lm, lm_region, rect):

    # Sizing up face region so that Laughing Man covers the whole face.
    rect[0] = max(rect[0] - 30, 0)
    rect[1] = max(rect[1] - 30, 0)
    rect[2] = min(rect[2] + 60, frame.shape[1] - rect[0])
    rect[3] = min(rect[3] + 60, frame.shape[0] - rect[1])

    segmented_lm = cv2.resize(segmented_lm, tuple(rect[2:]))
    lm_region = cv2.resize(lm_region, tuple(rect[2:]))

    masked_frame = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :] * (1.0 - lm_region)

    frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :] = segmented_lm + masked_frame

    return frame


def stream_lm_gif(gif_cap):

    gif_cap.grab()
    gif_cap.grab()
    gif_ret, lm = gif_cap.read()
    if not gif_ret:
        gif_cap = cv2.VideoCapture("laughing_man.gif")
        gif_ret, lm = gif_cap.read()

    return lm, gif_cap


def main():

    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # VideoCapture with camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    # VideoCapture from gif file
    gif_cap = cv2.VideoCapture("laughing_man.gif")
    _, lm = gif_cap.read()
    segmented_lm, mask = segment_lm(lm)

    # Main loop
    while ret:

        # Face detection with small-sizing acceleration.
        small_frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4)))
        faces = np.array(cascade.detectMultiScale(small_frame)) * 4

        # Synthesize Laughing Man
        if faces.shape[0] > 0:

            for rect in faces:
                frame = synth_lm(frame, segmented_lm, mask, rect)

        cv2.imshow('laughing man', frame)

        ret, frame = cap.read()

        lm, gif_cap = stream_lm_gif(gif_cap)
        segmented_lm = lm * mask

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

