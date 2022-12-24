import cv2
import numpy as np
import deep_sort.tracker
import deep_sort.detection
import core.utils as utils
import matplotlib.pyplot as plt
from core.config import cfg
from tools import generate_detections as gdet
from deep_sort import preprocessing, nn_matching

np.random.seed(8)


def drawBoundingBox(frame, box, text, color):
    text = str(text)
    color = [i * 255 for i in color]
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0]) + len(text) * 11 + 7, int(box[1] + 21)), color, -1)
    cv2.putText(frame, text, (int(box[0]) + 3, int(box[1] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


class Tracker:
    
    def __init__(self):
        self.nms_max_overlap = 1.0
        self.info = False
        
        metric_type = 'cosine'  # 'cosine' or 'euclidean'
        max_distance = 100000
        metric = nn_matching.NearestNeighborDistanceMetric(metric_type, max_distance)
        self.encoder = gdet.create_box_encoder('./yolov4-deepsort/model_data/mars.pb', batch_size=1)
        
        self.tracker = deep_sort.tracker.Tracker(metric, max_iou_distance=10, max_age=50, n_init=5)
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        
        color_map = plt.get_cmap('tab20b')
        self.colors = [color_map(i)[:3] for i in np.linspace(0, 1, 20)]
    
    def run(self, frame, bboxes, scores, names):
        # encode yolo detections and feed to tracker
        features = self.encoder(frame, bboxes)
        detections = [deep_sort.detection.Detection(bbox, score, name, feature) for
                      bbox, score, name, feature in
                      zip(bboxes, scores, names, features)]
        
        # run non-maxima supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxes, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # print(features)
        # for d in detections:
        #     print(vars(d))
        
        # Call the tracker
        self.tracker.predict()
        self.tracker.update(detections)
        
        tracks_num = 0
        
        out = frame.copy()
        # update tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = self.class_names[track.get_class()]
            tracks_num += 1
            
            drawBoundingBox(out, bbox, track.track_id, self.colors[int(track.track_id) % len(self.colors)])
            
            # if enable info flag then print details about each track
            if self.info:
                print('Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}'.format(
                    str(track.track_id),
                    class_name,
                    (int(bbox[0]),
                     int(bbox[1]),
                     int(bbox[2]),
                     int(bbox[3]))
                ))
        
        print(len(names), tracks_num, len(detections), '--' * tracks_num)
        cv2.imshow('tracker', out)
