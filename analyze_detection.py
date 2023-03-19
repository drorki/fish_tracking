'''
Performance evaluation for the fish detector, by estimating the probability of detection (PD) and probability of false
alarm (PFA) at each frame.
PD is estimated by the number of fish detected at each frame, divided by the number of fish. Mean PD is the mean over
all frames in which tracking occur (after initialization).
PFA is estimated by the number of trackers that are not near a fish, divided by the total number of trackers at that
frame. This is calculated only for frames in which trackers exist.
The relative coverage for each fish is the number of frames that it was covered by a tracker, divided by the total
number of frames.
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os

def convert(size,x,y,w,h):
    box = np.zeros(4)
    dw = 1./size[0]
    dh = 1./size[1]
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    box[0] = x-(w/2.0)
    box[1] = x+(w/2.0)
    box[2] = y-(h/2.0)
    box[3] = y+(h/2.0)
    return (box)


def get_annotated_track(data_path, img_h, img_w):
    '''
    In yolo, a bounding box is represented by four values [x_center, y_center, width, height] . x_center and y_center
    are the normalized coordinates of the center of the bounding box. To make coordinates normalized, we take pixel
    values of x and y, which marks the center of the bounding box on the x- and y-axis
    '''
    #n_frames = 816
    annotation_files = glob.glob(os.path.join(data_path, 'frame*.txt'))
    n_frames = len(annotation_files)
    bbox = []
    xy = []
    for frame_no in range(n_frames):  # Iterate over frame_no instead of filename, since we don't know the order of the files in annotation_files
        #filename = data_path + r'\frame' + str(frame_no) + '.txt'
        filename = os.path.join(data_path, 'frame' + str(frame_no) + '.txt')
        txt_file = open(filename, "r")
        line = txt_file.read().splitlines()
        values = line[0].split()
        x = y = w = h = cls = None
        cls = line[0]
        x = float(values[1])
        y = float(values[2])
        w = float(values[3])
        h = float(values[4])
        bbox.append(convert((img_w, img_h), x, y, w, h))
        xy.append([x, y])
    return xy, bbox


def relative2pixels(xy_list, img_w, img_h):
    # convert relative coordinates in xy_list to pixels
    xy = np.array(xy_list)
    xy[:, 0] = xy[:, 0] * img_w  # x-coordinate in pixels
    xy[:, 1] = xy[:, 1] * img_h  # y-coordinate in pixels
    return xy

############## PARAMETERS ################################
img_h = 1296  # pixels
img_w = 2304  # pixels
dist_thd_pixels = 50
video_path = r'.\data\video'

video_of_clean_container = True  # Determine which annotation to use: original (=False) of clean container (=True)

if video_of_clean_container:
    # ---- For new video with a clean container, e.g., VIDEO_20230305_111158706 ------------------------
    annotation_paths = [                                        # List of directories with annotation files
        '/VIDEO_20230305_111158706_annotation/fish_bottom_left',
        '/VIDEO_20230305_111158706_annotation/fish_left_middle',
        '/VIDEO_20230305_111158706_annotation/fish_left_top',
        '/VIDEO_20230305_111158706_annotation/fish_right_bottom',
        '/VIDEO_20230305_111158706_annotation/fish_right_middle',
        '/VIDEO_20230305_111158706_annotation/fish_right_top'
        ]
    filename2load = 'tracks_20230310-100823.pkl'  # A Model for train video with clean container, trained on 40 images over 9000 epochs
else:
    # ---- For original video, e.g., VIDEO_20230223_133606599 ------------------------
    annotation_paths = [                                           # List of directories with annotation files
        '/VIDEO_20230223_133606599_annotation/fish_bottom_right',
        '/VIDEO_20230223_133606599_annotation/fish_first_from_right',
        '/VIDEO_20230223_133606599_annotation/fish_first_left',
        '/VIDEO_20230223_133606599_annotation/fish_second_from_left',
        '/VIDEO_20230223_133606599_annotation/fish_second_from_right',    # 'Original' annotation
        '/VIDEO_20230223_133606599_annotation/fish_third_from_right'
        ]

    #filename2load = 'tracks_20230228-173213.pkl'  # max_iou_dist = 0.5, max_age = 12, n_init=4, max_cosine_distance = 0.4 --> 147/816
    #filename2load = 'tracks_20230228-181308.pkl'  # max_iou_dist = 0.2, max_age = 12, n_init=4, max_cosine_distance = 0.4 --> very few trackers
    #filename2load = 'tracks_20230228-181822.pkl'  # max_iou_dist = 0.8, max_age = 12, n_init=4, max_cosine_distance = 0.4 --> 183/816
    #filename2load = 'tracks_20230228-183435.pkl'  # max_iou_dist = 0.8, max_age = 12, n_init=4, max_cosine_distance = 0.8 --> 184/816
    #filename2load = 'tracks_20230301-122144.pkl'  # ROI, 10000 epochs, max_cosine_distance = 0.4, max_age = 12, max_iou_dist = 0.5, n_init=4, nn_budget=None <-- 241 / 816
    #filename2load = 'tracks_20230301-170515.pkl'  # ROI, 10000 epochs, max_cosine_distance = 0.4, max_age = 12, max_iou_dist = 0.5, n_init=3, nn_budget=None <--  282 / 816
    #filename2load = 'tracks_20230301-171343.pkl'  # ROI, 10000 epochs, max_cosine_distance = 0.9, max_age = 12, max_iou_dist = 0.5, n_init=3, nn_budget=None <-- 293 / 816
    #filename2load = 'tracks_20230301-172726.pkl'  # ROI, 10000 epochs, max_cosine_distance = 0.9, max_age = 12, max_iou_dist = 0.5, n_init=3, nn_budget=1 <-- 293 / 816
    #filename2load = 'tracks_20230301-173333.pkl'  # ROI, 10000 epochs, max_cosine_distance = 0.9, max_age = 12, max_iou_dist = 0.8, n_init=3, nn_budget=None <-- 307 / 816
    #filename2load = 'tracks_20230301-174037.pkl'  # ROI, 10000 epochs, max_cosine_distance = 0.9, max_age = 12, max_iou_dist = 0.8, n_init=6, nn_budget=None <-- 186 / 816
    #filename2load = 'tracks_20230302-113337.pkl'  # ROI, 9000 epochs, max_cosine_distance = 0.9, max_age = 12, max_iou_dist = 0.8, n_init=3, nn_budget=None <--  483 / 816 (59%!)
    #filename2load = 'tracks_20230302-114112.pkl'  # ROI, 9000 epochs, max_cosine_distance = 0.9, max_age = 100, max_iou_dist = 0.8, n_init=3, nn_budget=None <-- 576 / 816 looks like too-many irrelevant tracks
    #filename2load = 'tracks_20230302-115356.pkl'  # ROI, 9000 epochs, max_cosine_distance = 0.9, max_age = 24, max_iou_dist = 0.8, n_init=3, nn_budget=None <--  514 / 816 (63%!)
    filename2load = 'tracks_20230302-120735.pkl'  # ROI, 9000 epochs, max_cosine_distance = 0.9, max_age = 24, max_iou_dist = 0.8, n_init=3, nn_budget=None, verify_detections <--  526 / 816 (64%!)
    #filename2load = 'tracks_20230302-122106.pkl'  # ROI, 15000 epochs, max_cosine_distance = 0.9, max_age = 24, max_iou_dist = 0.8, n_init=3, nn_budget=None, verify_detections <-- 515 / 816 (63%!)


# ------------------------------------------


if __name__ == '__main__':

    # Tracking data
    with open('performance_evaluation/' + filename2load, 'rb') as f:
        history_dict, max_cosine_distance, nn_budget, max_age, max_iou_dist, n_init, roi, height, width, model_path = pickle.load(f)
    print('\nFile:', filename2load, ' ---------------')
    print('model_path =', model_path)
    print('max_cosine_distance =', max_cosine_distance)
    print('nn_budget =', nn_budget)
    print('max_age =', max_age)
    print('max_iou_dist =', max_iou_dist)
    print('n_init =', n_init)
    print('Number of valid trackers:', len(history_dict.keys()))
    print('Available trackers are:', history_dict.keys())

    # Annotation data
    for fish_ind in range(len(annotation_paths)):
        #print('annotations_path =', annotation_paths[fish_ind])
        xy_list, bbox_list = get_annotated_track(video_path + annotation_paths[fish_ind], img_h, img_w)
        if fish_ind == 0:
            xy = np.zeros((len(annotation_paths), len(xy_list), 2))  # [fish_ind, frame_no, x/y]
        xy[fish_ind, :, :] = (relative2pixels(xy_list, img_w, img_h))
        xy[fish_ind, :, 0] = xy[fish_ind, :, 0] - roi['top left xy'][0]  # compensate x for the top-left of teh ROI
        xy[fish_ind, :, 1] = xy[fish_ind, :, 1] - roi['top left xy'][1]  # compensate y for the top-left of teh ROI

# Estimate PD: iterate over annotated traces, and count how many traces are covered by a tracker at each frame.
    n_frames = xy.shape[1]  # Number of frames
    n_fish = xy.shape[0]
    n_tracks = np.zeros(n_frames)  # Count number of tracks at each frame
    fa_count = np.zeros(n_frames)  # Count number of false alarms at each frame
    detections_count = np.zeros((n_fish, n_frames))  # count number of detections for each fish at each frame
    for frame_no in range(n_frames):
        for track_no in history_dict.keys():
            # If tracker was active in the current frame - find the fish it follows
            ind = np.array(history_dict[track_no]['frame']) == frame_no
            if any(ind):
                loc = np.array(np.array(history_dict[track_no]['center'])[ind,:])
                n_tracks[frame_no] += 1
                for fish_ind in range(n_fish):
                    # dist = np.abs(xy[frame, :] - xy_data)
                    dist = np.abs(xy[fish_ind, frame_no, :] - loc )
                    if np.bitwise_and(dist[:, 0] < dist_thd_pixels, dist[:, 1] < dist_thd_pixels):
                        detections_count[fish_ind, frame_no] += 1
                if np.sum(detections_count[:, frame_no])==0:  # if there are no detections
                    fa_count[frame_no] += 1

    n_detections = np.sum(detections_count, axis=0)  # Number of true detections at each frame (a single fish may have more than one tracker)
    n_detected = np.sum(detections_count > 0, axis=0)  # Number of detected trackers at each frame
    valid_frames = n_tracks > 0  # Consider only frames that has trackers.
    pfa = np.mean(np.divide(fa_count[valid_frames], n_tracks[valid_frames]))  # Mean probability of false alarm
    pd = np.mean(n_detected[n_init:] / n_fish)  # Mean probability of detection. The first n_init frames don't have detections since trackers are not initialized
    print('Probability of detection:', pd)
    print('Probability of false alarm:', pfa)

    # Calculate coverage for each fish
    coverage = np.sum(detections_count[:,n_init:] > 0, axis=1)/detections_count[:,n_init:].shape[1]  # Relative coverage for each fish
    for fish_ind in range(len(annotation_paths)):
        print('Relative coverage for', annotation_paths[fish_ind].split('\\')[-1], 'is', coverage[fish_ind])

    plt.figure
    plt.plot(n_tracks, label='number of tracks')
    plt.plot(fa_count, label='false alarms')
    plt.plot(n_detections, label='number of detections')
    plt.plot(n_detected, label='number of detected fish')
    #plt.plot(n_detections+fa_count, label='detections+FA')  # sanity check: n_detections+fa_count - n_tracks = 0
    plt.xlabel('frame number')
    plt.legend()
    plt.show()