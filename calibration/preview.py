#! /usr/bin/env python3
"""
A script to calibrate the PiCam module using a Charuco board

- Change CAMERA_RESOLUTION to a lower resolution to improve
  frame rate
"""

import time
import argparse

import cv2
import numpy as np

from charuco import charucoBoard
from imutils.video import VideoStream

from fullscreen.fullscreen import *


CAMERA_RESOLUTION = (1920, 1088)
SCREEN_RESOLUTION = (1920, 1080)
QUIT_KEY = 'q'


def show_fullscreen_image(frame, name='Fullscreen Preview'):
    """
    Given an image, display the image in full screen.
    Use Case:
        > Display camera charuco pattern with projector.
        > Display camera preview
    """
    screen = FullScreen(0)
    screen.imshow(frame)


def show_windowed_image(frame, name='Preview'):
    """
    Given an image, display the image in a window
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(name, frame)


def hide_fullscreen_image(name='Preview'):
    """
    Kill a named window
    """
    cv2.destroyWindow(name)


def preview_camera():
    """
    Display output of the camera
    """
    stream = cv2.VideoCapture(2) # External web camera
    time.sleep(2)  # needed to allow camera to boot
    while True:
        cap_success, frame = stream.read()
        show_windowed_image(frame, 'Camera Preview')
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break


def preview_charuco():
    """
    Display an image until user presses "q" to quit
    """
    charuco = charucoBoard.draw(SCREEN_RESOLUTION)
    show_fullscreen_image(charuco, 'Calibration Board')
    while True:
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break
    cv2.destroyAllWindows()


def preview_perspective(img_resolution=(1920, 1080)):
    width, height = img_resolution
    img = np.ones((height, width, 1), np.uint8) * 255
    show_fullscreen_image(img)
    while True:
        if cv2.waitKey(1) & 255 == ord(QUIT_KEY):
            break
    cv2.destroyAllWindows()


def main():
    """
    Handle parsing of arguments
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-c", "--camera", action="store_true")
    group.add_argument("-b", "--board", action="store_true")
    group.add_argument("-p", "--perspective", action="store_true")
    args = parser.parse_args()
    if args.camera:
        preview_camera()
    elif args.board:
        preview_charuco()
    elif args.perspective:
        preview_perspective()
    else:
        preview_camera()


if __name__ == "__main__":
    main()
