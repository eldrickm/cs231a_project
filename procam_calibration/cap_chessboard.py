"""
Capture projection pattern and decode x-coorde.
"""
import cv2
import numpy as np
import os
import fullscreen.fullscreen as fs

CAMERA_RESOLUTION = (1920, 1080)
VIDEO_BUFFER_LEN = 4

def flush_cap(cap):
    """
    An attempt to flush the video capture buffer.
    Needed because the first VIDEO_BUFFER_LEN frames of a series of captures
    essentially won't update properly / will be stale.
    This is unfortunately a hardware issue and can't be easily disabled
    """
    for i in range(5):
        cap.grab()


def imshowAndCapture(cap, img_pattern, screen, delay=500):
    screen.imshow(img_pattern)
    cv2.waitKey(delay)
    ret, img_frame = cap.read()
    cv2.waitKey(delay)
    img_frame = cv2.resize(img_frame, CAMERA_RESOLUTION)
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    return img_gray


def main():
    cap = cv2.VideoCapture(0) # External web camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
   
    # Get graycode pattern files
    images = [None] * len(os.listdir('./graycode_pattern/'))

    for filename in os.listdir('./graycode_pattern/'):
        if filename.endswith(".png"): 
            image = cv2.imread('./graycode_pattern/'+filename)
            position = int(filename[8:10])
            images[position] = image
        else:
            continue

    for i in range(VIDEO_BUFFER_LEN):
        images.append(images[-1])

    # Capture
    screen = fs.FullScreen(0)
    black = np.zeros(CAMERA_RESOLUTION)
    screen.imshow(black)
    flush_cap(cap)
    imlist = [imshowAndCapture(cap, img, screen) for img in images]

    # Chop off lagging frames
    imlist = imlist[VIDEO_BUFFER_LEN:]

    # Save files
    for index, img in enumerate(imlist):
        if(index < 10):
            str_ind = '0' + str(index)
        else:
            str_ind = str(index)

        cv2.imwrite("./captures/capture_0/graycode_" + str_ind + ".png", img)
    cv2.destroyAllWindows()
    cap.release()

if __name__=="__main__":
    main()
