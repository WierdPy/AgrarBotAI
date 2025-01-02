import sys

from yolov9.ExtendedDetections import ExtendedDetections

if 'yolov9' not in sys.path:
    sys.path.append('yolov9')

# ML/DL
import numpy as np
import torch

# CV
import cv2
import supervision as sv

# YOLOv9
from models.common import DetectMultiBackend, AutoShape
from utils.general import set_logging

# Video Demonstration
from IPython.display import HTML
from base64 import b64encode

from supervision import Detections as BaseDetections
from supervision.config import CLASS_NAME_DATA_FIELD


def setup_model_and_video_info(model, config, source_path):
    # Initialize and configure YOLOv9 model
    model = prepare_yolov9(model, **config)
    # Retrieve video information
    video_info = sv.VideoInfo.from_video_path(source_path)
    return model, video_info


def prepare_yolov9(model, conf=0.2, iou=0.7, classes=None, agnostic_nms=False, max_det=1000):
    model.conf = conf
    model.iou = iou
    model.classes = classes
    model.agnostic = agnostic_nms
    model.max_det = max_det
    return model


def create_byte_tracker(video_info):
    # Setup BYTETracker with video information
    return sv.ByteTrack(track_thresh=0.25, track_buffer=250, match_thresh=0.95, frame_rate=video_info.fps)


def setup_annotators():
    c = sv.ColorLookup.TRACK  # Colorize based on the TRACK id, as opposed to INDEX or CLASS
    # Initialize various annotators for bounding boxes, traces, and labels
    bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2, color_lookup=c)
    round_box_annotator = sv.RoundBoxAnnotator(thickness=2, color_lookup=c)
    corner_annotator = sv.BoxCornerAnnotator(thickness=2, color_lookup=c)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50, color_lookup=c)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, color_lookup=c)
    return [bounding_box_annotator, round_box_annotator, corner_annotator], trace_annotator, label_annotator


def setup_counting_zone(counting_zone, video_info):
    # Configure counting zone based on provided parameters
    max_width = video_info.width - 1
    max_height = video_info.height - 1
    if counting_zone == 'whole_frame':
        polygon = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]])
    else:
        polygon = np.clip(counting_zone, a_min=[0, 0], a_max=[max_width, max_height])
    polygon_zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(video_info.width, video_info.height),
                                  triggering_position=sv.Position.CENTER)
    polygon_zone_annotator = sv.PolygonZoneAnnotator(polygon_zone, sv.Color.ROBOFLOW,
                                                     thickness=4 * (2 if counting_zone == 'whole_frame' else 1),
                                                     text_thickness=2, text_scale=2)
    return polygon_zone, polygon_zone_annotator


def annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone,
                   polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels, model):
    # Apply tracking to detections
    detections = byte_tracker.update_with_detections(detections)
    annotated_frame = frame.copy()

    # Handle counting zone logic
    if counting_zone is not None:
        is_inside_polygon = polygon_zone.trigger(detections)
        detections = detections[is_inside_polygon]
        annotated_frame = polygon_zone_annotator.annotate(annotated_frame)

    # Annotate frame with traces
    annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)

    # Annotate frame with various bounding boxes
    section_index = int(index / (video_info.total_frames / len(annotators_list)))
    annotated_frame = annotators_list[section_index].annotate(scene=annotated_frame, detections=detections)

    # Optionally, add labels to the annotations
    if show_labels:
        annotated_frame = add_labels_to_frame(label_annotator, annotated_frame, detections, model)

    return annotated_frame


def add_labels_to_frame(annotator, frame, detections, model):
    labels = [f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}" for confidence, class_id, tracker_id in
              zip(detections.confidence, detections.class_id, detections.tracker_id)]
    return annotator.annotate(scene=frame, detections=detections, labels=labels)


def process_video(model, config=dict(conf=0.1, iou=0.45, classes=None, ), counting_zone=None, show_labels=False,
                  source_path='input.mp4', target_path='output.mp4'):
    model, video_info = setup_model_and_video_info(model, config, source_path)
    byte_tracker = create_byte_tracker(video_info)
    annotators_list, trace_annotator, label_annotator = setup_annotators()
    polygon_zone, polygon_zone_annotator = setup_counting_zone(counting_zone, video_info) if counting_zone else (
        None, None)

    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        frame_rgb = frame[..., ::-1]  # Convert BGR to RGB
        results = model(frame_rgb, size=(video_info.height, video_info.width), augment=False)
        detections = ExtendedDetections.from_yolov9(results)
        return annotate_frame(frame, index, video_info, detections, byte_tracker, counting_zone, polygon_zone,
                              polygon_zone_annotator, trace_annotator, annotators_list, label_annotator, show_labels,
                              model)

    sv.process_video(source_path=source_path, target_path=target_path, callback=callback)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights='yolov9-e-converted.pt', device=device, data='data\coco.yaml', fuse=True)
model = AutoShape(model)

process_video(
    model,
    config=dict(conf=0.2, iou=0.6, classes=0),
    counting_zone='whole_frame',
    show_labels=True,
    source_path='Videos\crowd.mp4',
    target_path='full_frame_detection_tracking_counting_yolov9.mp4'
)
