
import numpy as np
import matplotlib.pyplot as plt
import pickle


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
    n_frames = 816
    bbox = []
    xy = []
    for frame_no in range(n_frames):
        filename = data_path + r'\frame' + str(frame_no) + '.txt'
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
#video_path = r'C:\Users\d_kip\work\lab_projects\Fish_Noise\Video\VIDEO_20230223_133606599'
annotations_path = r'\VIDEO_20230223_133606599_annotation\fish_bottom_right'
#annotations_path = video_path + r'\fish_first_from_right'
#annotations_path = video_path + r'\fish_first_left'
#annotations_path = video_path + r'\fish_second_from_left'
#annotations_path = video_path + r'\fish_second_from_right' # 'Original' annotation
#annotations_path = video_path + r'\fish_third_from_right'

###########################################################


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
#filename2load = 'tracks_20230302-120735.pkl'  # ROI, 9000 epochs, max_cosine_distance = 0.9, max_age = 24, max_iou_dist = 0.8, n_init=3, nn_budget=None, verify_detections <--  526 / 816 (64%!)
filename2load = 'tracks_20230302-122106.pkl'  # ROI, 15000 epochs, max_cosine_distance = 0.9, max_age = 24, max_iou_dist = 0.8, n_init=3, nn_budget=None, verify_detections <-- 515 / 816 (63%!)
# ------------------------------------------

if __name__ == '__main__':

    # Tracking data
    with open('performance_evaluation/' + filename2load, 'rb') as f:
        #history_dict, nn_budget, max_cosine_distance, max_age, max_iou_dist, frame_height, frame_width = pickle.load(f)
        history_dict, max_cosine_distance, nn_budget, max_age, max_iou_dist, n_init, roi, frame_height, frame_width, model_path = pickle.load(f)
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
    print('annotations_path =', annotations_path)
    xy_list, bbox_list = get_annotated_track(video_path + annotations_path, img_h, img_w)
    xy = relative2pixels(xy_list, img_w, img_h)
    xy[:, 0] = xy[:, 0] - roi['top left xy'][0]  # compensate x for the top-left of teh ROI
    xy[:, 1] = xy[:, 1] - roi['top left xy'][1]  # compensate y for the top-left of teh ROI


    n_coverage_frames = 0
    good_tracks_count = 0  # Count number of tracks that follows the target at least 60% of their duration
    # Calculate coverage
    for key in history_dict.keys():
        xy_data = np.array(history_dict[key]['center'])
        frame = np.array(history_dict[key]['frame'])-2
        dist = np.abs(xy[frame, :] - xy_data)
        n_coverage_track = np.sum(np.bitwise_and(dist[:, 0] < dist_thd_pixels, dist[:, 1] < dist_thd_pixels))  # Count the number of frames during which the current track follows the fish
        good_tracks_count += n_coverage_track/xy_data.shape[0] > 0.6  # We consider a track as 'good' if at least 60% of it follows the fish
        print('Coverage of track', key, 'is', n_coverage_track, '/', xy_data.shape[0])
        n_coverage_frames += n_coverage_track
    print('dist_thd_pixels =', dist_thd_pixels, '--> n_coverage_frames =', n_coverage_frames, '/', xy.shape[0])
    print('Number of tracks which follows the fish at least 60% of the time:', good_tracks_count)
    # Plot x vs. frame number
    plt.figure().canvas.manager.set_window_title(annotations_path.split('\\')[-1])
    plt.plot(xy[:,0], '--')
    plt.fill_between(np.arange(len(xy)), xy[:,0]-dist_thd_pixels, xy[:,0]+dist_thd_pixels, color='b', alpha=.1)
    for key in history_dict.keys():
        xy_data = np.array(history_dict[key]['center'])
        frame = np.array(history_dict[key]['frame'])
        plt.plot(frame, xy_data[:,0], label=str(key))
    plt.xlabel('frame#')
    plt.ylabel('x')

    # Plot y vs. frame number
    plt.figure().canvas.manager.set_window_title(annotations_path.split('\\')[-1])
    plt.plot(xy[:,1], '--')
    plt.fill_between(np.arange(len(xy)), xy[:, 1] - dist_thd_pixels, xy[:, 1] + dist_thd_pixels, color='b', alpha=.1)
    for key in history_dict.keys():
        xy_data = np.array(history_dict[key]['center'])
        frame = np.array(history_dict[key]['frame'])
        plt.plot(frame, xy_data[:,1], label=str(key))
    plt.xlabel('frame#')
    plt.ylabel('y')

    plt.show()

'''
    # Plot (x,y)
    plt.figure().canvas.manager.set_window_title(annotations_path.split('\\')[-1])
    plt.plot(xy[:,0], xy[:,1], '.-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
'''