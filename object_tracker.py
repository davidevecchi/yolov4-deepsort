from absl import app
import core.utils as utils
from core.config import cfg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import time


# Definition of the parameters
max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 0.0
info = False

# initialize deep sort
model_filename = 'yolov4-deepsort/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)
class_names = utils.read_class_names(cfg.YOLO.CLASSES)


def track(frame, bboxes, scores, names):
    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(bboxes, scores, names, features)]
    
    # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    
    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    
    print(len(names), len(tracker.tracks))
    
    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = class_names[track.get_class()]
        
        # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                      (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 11, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)),
                    0, 0.5, (255, 255, 255), 2)
        
        # if enable info flag then print details about each track
        if info:
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                str(track.track_id),
                class_name,
                (int(bbox[0]),
                 int(bbox[1]),
                 int(bbox[2]),
                 int(bbox[3]))
            ))
    
    cv2.imshow('tracker', frame)


def main(_argv):
    # while True:
    #     return_value, frame = vid.read()
    #     if return_value:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     else:
    #         print('Video has ended or failed, try a different video format!')
    #         break
    #     track(frame)
    pass


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
