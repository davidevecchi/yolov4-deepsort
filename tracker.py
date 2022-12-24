import cv2
import numpy as np
import deep_sort.tracker
import deep_sort.detection
import core.utils as utils
import matplotlib.pyplot as plt
from core.config import cfg
from tools import generate_detections as gdet
from deep_sort import preprocessing, nn_matching


class Tracker:
    
    def __init__(self):
        self.max_cosine_distance = 1
        self.nms_max_overlap = 1
        self.info = False
        
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', self.max_cosine_distance)
        self.encoder = gdet.create_box_encoder('./yolov4-deepsort/model_data/market1501.pb', batch_size=1)
        self.tracker = deep_sort.tracker.Tracker(metric)
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
            
            self.drawBoundingBox(out, track, bbox, class_name)
            
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
        
        print(len(names), tracks_num, '--' * tracks_num)
        cv2.imshow('tracker', out)
    
    def drawBoundingBox(self, frame, track, bbox, class_name):
        color = self.colors[int(track.track_id) % len(self.colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
        #               (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 11, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + '-' + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
