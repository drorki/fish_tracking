# An article on deepSORT can be found here:
# https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104


import os
# comment out below line to enable tensorflow logging outputs
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
import pickle  # To save results into a file
#from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat  # to save mat files
from tensorflow.compat.v1 import ConfigProto


# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


################################################


def detect(interpreter, input_tensor):
    """Run detection on an input image.

    Args:
      interpreter: tf.lite.Interpreter
      input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.

    Returns:
      A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
    """

    # Detection function for TFLite model
    # output_details[0]['index'] --> scores
    # output_details[1]['index'] --> boxes
    # output_details[3]['index'] --> classes

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(0, [1, 320, 320, 3])  # input_tensor.shape = TensorShape([1, 416, 416, 3])
    interpreter.allocate_tensors()
    resized_tensor = tf.image.resize(input_tensor, [320, 320])
    interpreter.set_tensor(input_details[0]['index'], resized_tensor.numpy())
    interpreter.invoke()  # Be sure to set the input sizes, allocate tensors and fill values before calling this.

    scores = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    #n_detections = interpreter.get_tensor(output_details[2]['index'])
    classes = interpreter.get_tensor(output_details[3]['index'])

    detections = {'boxes': boxes, 'scores': scores, 'classes': classes}
    return detections

# ---- Utilities ------------------------
def plot_track_data(history_dict, track_no, frame_height, frame_width):
    xy_data = np.array(history_dict[track_no]['center'])
    time_data = np.array(history_dict[track_no]['timestamp'])
    frame_data = np.array(history_dict[track_no]['frame'])
    width = np.array(history_dict[track_no]['width'])
    height = np.array(history_dict[track_no]['height'])

    fig, axs = plt.subplots(2, 2)
    fig.canvas.manager.set_window_title('Track #' + str(track_no))

    # Plot (x,y) coordinates
    axs[0,0].plot(xy_data[:,0], xy_data[:,1], marker='x')
    axs[0,0].set_xlabel('x')
    axs[0,0].set_ylabel('y')
    axs[0,0].set_xlim(1, frame_width)
    axs[0,0].set_ylim(1, frame_height)
    axs[0,0].invert_yaxis()

    # Plot x(frame no.) and y(frame no.)
    axs[0,1].plot(frame_data, xy_data[:,0], 'x', label='x')
    axs[0,1].plot(frame_data, xy_data[:,1], '.', label='y')
    axs[0,1].set_xlabel('frame no.')
    axs[0,1].legend()

    # Plot width(t) and height(t)
    axs[1, 0].plot(time_data / 1000, width, 'x', label='width')
    axs[1, 0].plot(time_data / 1000, height, '.', label='height')
    axs[1, 0].set_xlabel('time [s]')
    axs[1, 0].set_ylabel('pixels')
    axs[1, 0].legend()

    # Plot x(t) and y(t)
    axs[1,1].plot(time_data/1000, xy_data[:,0], 'x', label='x')
    axs[1,1].plot(time_data/1000, xy_data[:,1], '.', label='y')
    axs[1,1].set_xlabel('time [s]')
    axs[1,1].legend()

    plt.show()


def verify_detections(detections, roi):
    boxes = np.array([d.tlwh for d in detections])  # TL = (int(bbox[0]), int(bbox[1])), BR=(int(bbox[2]), int(bbox[3]))
    detections_verified = []
    for i in range(boxes.shape[0]):
        ok_flag = (boxes[i, :][0] >= roi['top left xy'][0]) & \
                (boxes[i, :][1] >= roi['top left xy'][1]) & \
                (boxes[i, :][2] <= roi['bottom right xy'][0]) & \
                (boxes[i, :][3] <= roi['bottom right xy'][1])
        if ok_flag:
            detections_verified.append(detections[i])
    return detections_verified


def save_track_as_mat(tracker_no, history_dict, data2save, path):
# Save data of tracker number tracker_no, and additional data given in data2save, in to a mat file
    data = list(history_dict[tracker_no].items())
    for item in data:
        data2save[item[0]] = np.array(item[1])
        #print('item[0]=', item[0])
        #print('item[1]=', item[1])
    savemat(os.path.join(path, str(tracker_no) + '.mat'), data2save)


def main():

    ############## PARAMETERS ################################

    roi = {'top left xy': (900, 75), 'bottom right xy': (2000, 1175)}   # Region of interest in input video

    # Parameters for deepSORT tracking
    max_cosine_distance = 0.9  #0.4  # a threshold to determine the person similarity by ReID. The higher the value, the easier it is to assume it is the same person.
    nn_budget = None # a value that indicates how many previous frames of feature vectors should be retained for distance calculation for each track.
    max_age = 24 #12 #6  # Maximal allowed age for a track (A_max parameter in deepSORT paper)
    max_iou_dist = 0.8  # maximal IOU distance between bounding boxes
    n_init = 3
    nms_max_overlap = 1.0
    history_dict = {}

    iou_threshold = 0.45  # iou threshold for choosing detections by non-max suppression
    score_threshold = 0.50  # score threshold for choosing detections by non-max suppression

    # Detection model file
    model_path = 'data/model7_ROI.tflite'  # Model for train video with clean container, trained on 40 images over 9000 epochs
    #model_path = 'data/model6_ROI_Mar2_2023.tflite'  # Model for train video with ROI, trained on 40 images over 15000 epochs
    #model_path = 'data/model5_ROI_Mar2_2023.tflite'  # Model for train video with ROI, trained on 40 images over 9000 epochs
    #model_path = 'data/model4_ROI_Mar2_2023.tflite'  # Model for train video with ROI, trained on 20 images over 3000 epochs
    #model_path = 'data/model3_ROI_Mar1_2023.tflite'  # Model for new camera with ROI, trained on 40 images over 10000 epochs

    #video_path = './data/video/VIDEO_20230304_100716744.mp4'  # Path of input video
    video_path = './data/video/VIDEO_20230305_111158706.mp4'  # Path of input video: clean pool with annotation
    CUT_ROI_FLAG = True  # Cut a rectangle around ROI

    SHOW_VIDEO_FLAG = True  # True/False to show/don't show video output on screen
    PRINT_INFO_FLAG = False  # Print detailed info of tracked objects on screen
    COUNT_OBJECTS_FLAG = False  # Print count of objects being tracked on screen

    # --- CURRENTLY THIS FUNCTIONALITY IS NOT WORKING ----
    SAVE_VIDEO_FLAG = True  # True/False to save video output | NOTE: CURRENTLY THIS FUNCTIONALITY IS NOT WORKING
    video_output_path = './outputs/fish_square.AVI' # NOTE: CURRENTLY THIS FUNCTIONALITY IS NOT WORKING
    # -----------------------------------------

    ##################################################################################

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)  # use cosine distance metric

    # initialize tracker
    tracker = Tracker(metric, max_iou_distance=max_iou_dist, max_age=max_age, n_init=n_init)   # Default values: max_iou_distance=0.7, max_age=60, n_init=3

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    #video_path = FLAGS.video

    interpreter = tf.lite.Interpreter(model_path=model_path)  # Load the TFLite detection model and allocate tensors

    # begin video capture
    vid = cv2.VideoCapture(video_path)

    # get video ready to save locally if flag is set


    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if SAVE_VIDEO_FLAG:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_output_path, codec, fps, (width, height))  # Size of output is height x width
    else:
        out = None

    frame_num = 0  # frames counter
    frame_height = frame_width = None

    # while video is running
    while True:  # & (frame_num < 100):
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image = Image.fromarray(frame)  # TODO: original line. verify if needed
            timestamp = vid.get(cv2.CAP_PROP_POS_MSEC)  # time in milliseconds from the start of the video
            frame_height, frame_width = frame.shape[:2]
            frame_num += 1
            print('frame', frame_num)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        if CUT_ROI_FLAG:
            frame = frame[roi['top left xy'][1]:roi['bottom right xy'][1], roi['top left xy'][0]:roi['bottom right xy'][0]]
        '''
        ## Code to display a single frame     
        cv2.namedWindow("frame with ROI", cv2.WINDOW_NORMAL)
        top_left_xy = (900, 50)
        bottom_right_xy = (2000, 1200)
        image1 = cv2.rectangle(frame, top_left_xy, bottom_right_xy, (255, 0, 0), 2)  # frame.shape = (1296, 2304, 3)
        cv2.imshow('frame with ROI', image1)
        # cv2.imshow('frame with ROI', frame)
        cv2.waitKey(0)
        '''

        image_data = frame/255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite
        input_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
        detections = detect(interpreter, input_tensor)
        boxes = detections['boxes']   # Bounding boxes from detector
        batch_size = tf.shape(boxes)[0]
        num_boxes = boxes.shape[1]
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (batch_size, -1, 1, 4)),   # shape should be [batch_size, num_boxes, q, 4], where q=1
            scores=tf.reshape(detections['scores'], (batch_size, num_boxes, 1)),  # shape should be: [batch_size, num_boxes, num_classes]
            max_output_size_per_class=num_boxes,
            max_total_size=num_boxes,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        class_names = {0: 'fish'}
        allowed_classes = ['fish']
        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        if COUNT_OBJECTS_FLAG:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))

        # delete detections that are not in allowed_classes
        if len(deleted_indx):
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

        # encode detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        if CUT_ROI_FLAG:
            dummy_roi = {'top left xy': (1, 1),
                         'bottom right xy': (roi['bottom right xy'][0] - roi['top left xy'][0], roi['bottom right xy'][1] - roi['top left xy'][1])}
            detections = verify_detections(detections, dummy_roi)
        else:
            detections = verify_detections(detections, roi)  # verify that bbox is inside the ROI

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # print('scores =', scores)
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if PRINT_INFO_FLAG:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # Save history of tracks
            if not (track.track_id in history_dict):
            # Initialize a dictionary key for current tracker
                history_dict[track.track_id] = {'center':[], 'frame':[], 'class_name':[], 'timestamp':[], 'height':[], 'width':[]}
            top_left = bbox[:2]  # top_left = (bbox[0]), int(bbox[1])
            bottom_right = bbox[2:]  # bottom_right = (bbox[2]), int(bbox[3])
            history_dict[track.track_id]['center'].append((top_left + bottom_right)/2)
            history_dict[track.track_id]['frame'].append(frame_num)
            history_dict[track.track_id]['class_name'].append(class_name)
            history_dict[track.track_id]['timestamp'].append(timestamp)
            history_dict[track.track_id]['height'].append(bbox[2] - bbox[0])  # bbox = [TLy, TLx, BRy, BRx]
            history_dict[track.track_id]['width'].append(bbox[3] - bbox[1])

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if SHOW_VIDEO_FLAG:
            cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL)  # Allow window resize
            cv2.imshow("Output Video", result)
        
        # if flag is set, save video file
        if SAVE_VIDEO_FLAG:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vid.release()
    if SAVE_VIDEO_FLAG:
        out.release()
    cv2.destroyAllWindows()

    # ---- Save results to a file ----
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename2save = os.path.join('performance_evaluation', 'tracks_' + timestr + '.pkl')
    # Save history_dict
    print('Saving file:', filename2save)
    with open(filename2save, 'wb') as f:
        pickle.dump([history_dict, max_cosine_distance, nn_budget, max_age, max_iou_dist, n_init, roi, height, width, model_path], f)

    # Save to mat files: a file for each tracker
    if nn_budget == None:  # Patch to enable savemat
        nn_budget = -1
    data2save = {'max_cosine_distance': max_cosine_distance, 'nn_budget': nn_budget, 'max_age': max_age, 'max_iou_dist': max_iou_dist,
                 'n_init': n_init, 'roi': roi, 'height': height, 'width': width, 'model_path': model_path}
    path4mat = os.path.join('performance_evaluation', filename2save.split('.')[0] + '_mat')
    path2save_mat = filename2save.split('.')[0] + '_mat'
    os.mkdir(path2save_mat)
    for tracker_no in history_dict.keys():
        save_track_as_mat(tracker_no, history_dict, data2save, path2save_mat)

    # Plot history
    plt.figure()
    plt.imshow(result)

    for track_no in history_dict.keys():
        line, = plt.plot(*zip(*history_dict[track_no]['center']), marker='.')
        mean_loc = np.mean(history_dict[track_no]['center'], axis=0)
        plt.text(mean_loc[0], mean_loc[1], track_no, color=line.get_color(), weight='bold')
        #plt.text(mean_loc[0]-50, mean_loc[1]+20, set(history_dict[track_no]['class_name']), color=line.get_color(), weight='bold')
    plt.show()

    # ------- Print summary ---------------
    print('Number of valid trackers:', len(history_dict.keys()))
    print('Available trackers are:', history_dict.keys())


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
