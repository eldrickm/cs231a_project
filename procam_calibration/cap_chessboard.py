"""
Capture projection pattern and decode x-coorde.
"""
import cv2
import numpy as np
import os
import fullscreen.fullscreen as fs

CAMERA_RESOLUTION = (640 // 2, 360 // 2) 

def imshowAndCapture(cap, img_pattern, delay=250):
    screen = fs.FullScreen(0)
    screen.imshow(img_pattern)
    cv2.waitKey(delay)
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    return img_gray

def main():
    width  = 640
    height = 480

    cap = cv2.VideoCapture(0) # External web camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
   
    # Get graycode pattern files
    images = [None] * 42

    for filename in os.listdir('./graycode_pattern/'):
        if filename.endswith(".png"): 
            image = cv2.imread('./graycode_pattern/'+filename)
            position = int(filename[8:10])
            images[position] = image
        else:
            continue

    # Capture
    imlist = [ imshowAndCapture(cap, img) for img in images]

    # Save files
    for index, img in enumerate(imlist):
        if(index < 10):
            str_ind = '0' + str(index)
        else:
            str_ind = str(index)

        cv2.imwrite("./capture0/graycode_" + str_ind + ".png", img)
    cv2.destroyAllWindows()
    cap.release()

if __name__=="__main__":
    main()
