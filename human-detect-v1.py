# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
from playsound import playsound as ps
import numpy as np
import argparse
import imutils
import time
import cv2
import time as ti
from datetime import date, datetime, time
from threading import Thread
import sys

class PlaySound(Thread):
    def __init__(self, fn):
        Thread.__init__(self) 
        self.fn = fn

    def run(self):
        ps(self.fn)
        print("Finished playing sound file: ", self.fn)

class SaveImage(Thread):
    def __init__(self, frame, fn):
        Thread.__init__(self) 
        self.frame = frame
        self.fn = fn

    def run(self):
        cv2.imwrite(self.fn, self.frame)
        print("Finished writing image file: ", self.fn)

def Main():

    soundAlarm = True
    start = datetime.now()

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("Starting camera stream thread")
    fvs = FileVideoStream("rtsp://cam1.home.local:554/user=admin&password=&channel=1&stream=0./").start()
    ti.sleep(1.0)

    # start the FPS timer
    fps = FPS().start()

    while fvs.more():
        try:

            # Capture frame-by-fracvme
            frame = fvs.read()

            #Resize and greyscale for faster detection
            frame = imutils.resize(frame, width=min(400, frame.shape[1]))

            #Display image queue buffer length
            cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)	
                
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Detect human forms
            (boxes, weights) = hog.detectMultiScale(img_gray, winStride=(8,8), padding=(8,8), scale=1.05 )

            #Display boxes around the areas of interest 
            for i, (x, y, w, h) in enumerate(boxes):
                if weights[i] > 0.85:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow('Alert',frame)
                    st = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    fn = "/prg/alert-" + str(st) + ".jpg"
                    save = SaveImage(frame, fn)
                    save.start()
                
            # Display the resulting frame
            cv2.imshow('frame',frame)
                                          
            elapsed = (datetime.now() - start).seconds
            if elapsed > 10:
                soundAlarm = True

            fps.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except AttributeError:
            print("[ERROR] Unable to grab frame.")
            break

    #On program exit, release the capture
    fps.stop()
    # finally, close the window
    cv2.destroyAllWindows()
    fvs.stop()

    cv2.waitKey(1)

if __name__=='__main__':
    Main()
