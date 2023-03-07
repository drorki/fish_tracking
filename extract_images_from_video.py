'''
Extract frames from a video file
Set REDUCE_RATE_FLAG=True and CAPTURE_INTERVAL_SECONDS=1 to reduce the frame rate to e.g, 1 frame per second.
set CUT_ROI_FLAG and roi to capture a specific region of interest
'''

import argparse
import os

import cv2
print(cv2.__version__)

# ----- Parameters ------------------------------------

REDUCE_RATE_FLAG = True  # True to reduce capture rate. False to use original rate
CAPTURE_INTERVAL_SECONDS = 1  # Capture every second
CUT_ROI_FLAG = True  # If true the video is truncated around the ROI
roi = {'top left xy': (900, 75), 'bottom right xy': (2000, 1175)}  # Region Of Interest in video

video_path = r'.\data\video'
#video_file = 'VIDEO_20230223_133606599.mp4'  # Video that has annotations
#video_file = 'VIDEO_20230302_081012380.mp4'  # Video only for training
video_file = 'VIDEO_20230304_100716744.mp4'  # Video for training with a clean container
video_file = 'VIDEO_20230305_111158706.mp4'
# ------------------------------------------------------------------

dir4frames = os.path.join(video_path, video_file.split('.')[0])
if REDUCE_RATE_FLAG:
    dir4frames = dir4frames + '_' + str(CAPTURE_INTERVAL_SECONDS) + 'FPS'
if CUT_ROI_FLAG:
    dir4frames = dir4frames + '_ROI'
if not os.path.exists(dir4frames):
    os.makedirs(dir4frames)
pathIn = os.path.join(video_path, video_file)
pathOut = dir4frames

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        if REDUCE_RATE_FLAG:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000*CAPTURE_INTERVAL_SECONDS))    # Capture every 1000*CAPTURE_INTERVAL_SECONDS milliseconds
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        if CUT_ROI_FLAG:
            image = image[roi['top left xy'][1]:roi['bottom right xy'][1], roi['top left xy'][0]:roi['bottom right xy'][0]]
        cv2.imwrite(pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print('pathIn =', pathIn)
    print('pathOut=', pathOut)
    extractImages(pathIn, pathOut)