# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import cv2

import numpy as np


if __name__ == "__main__":
    cam = cv2.VideoCapture(-1)

    # Set up the detector with default parameters.
    detector = cv2.ORB_create()

    while True:
        OK, img = cam.read()
        if not OK:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect blobs.
        keypoints = detector.detect(img)

        # Draw detected blobs as red circles.
        visu = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("Keypoints", visu)

        ch = cv2.waitKey(1)
        if ch in (ord('q'), ord('Q')):
            break
        elif ch in (ord('d'), ord('D')):
            import ipdb
            ipdb.set_trace()
