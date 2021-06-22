import cv2
import os

input_vid_name = 'input.mp4'
output_frame_folder = 'output'

cap = cv2.VideoCapture(input_vid_name)
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not os.path.exists(output_frame_folder):
    os.makedirs(output_frame_folder)

print(vid_length)

for framenum in range(0, vid_length):
    #print(framenum)
    cap.set(cv2.CAP_PROP_FRAME_COUNT, framenum)
    ret, frame = cap.read()

    if ret is False:
        break

    # Image Processing
    cv2.imwrite(output_frame_folder + '/' + str(framenum).zfill(5) + '.jpg', frame)
    cv2.imshow('frame', frame)
   # k = cv2.waitKey(50) & 0xff
    k = cv2.waitKey(100)
    if k == 27:  # Escape (ESC)
        break

cap.release()
cv2.destroyAllWindows()

'''

from PIL import Image
import os, sys
import cv2
import numpy as numpy
import matplotlib.pyplot as plt
vidcap = cv2.VideoCapture('4K Traffic camera video - Low.mp4')

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*30.0)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join('Video_Frame_Low', str(count) + '.jpg'), image)     # save frame as JPG file
    return hasFrames

sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec) '''