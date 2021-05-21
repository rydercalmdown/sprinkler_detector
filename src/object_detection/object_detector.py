import numpy as np
import imutils
import time
import cv2
import os
from imutils.video import VideoStream, FPS
from object_detection.detection import Detection


class ObjectDetector():
    """Class for running yolov3-tiny implementation"""

    def __init__(self):
        self._set_up()

    def __del__(self):
        self._clean_up()

    def _set_up(self):
        """Sets up resources"""
        self._set_defaults()
        self._load_labels()
        self._load_model_to_memory()
        self._set_fps_limit()

    def _set_defaults(self):
        this_files_dir = os.path.dirname(os.path.realpath(__file__))
        self.network = None
        self.confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.5))
        self.non_maxima_supression_threshold = 0.3
        self.model_weights_path = os.path.join(this_files_dir, 'models/yolov3-tiny.weights')
        self.model_config_path = os.path.join(this_files_dir, 'models/yolov3-tiny.cfg')
        self.labels_path = os.path.join(this_files_dir, 'models/coco.names')
        self.fps_running = 0
        self.fps_start_time = time.time()
        self.fps_wait_seconds = None
        self.should_display_fps = bool(os.environ.get('DISPLAY_FPS', False))

    def _calculate_frames_per_second(self):
        """Caculates frames per second"""
        update_every_x_seconds = 5.0
        current_time = time.time()
        time_difference = current_time - self.fps_start_time
        if time_difference > update_every_x_seconds:
            fps_actual = int(float(self.fps_running) / time_difference)
            print('FPS: {} - {}'.format(fps_actual, int(time.time())))
            self.fps_start_time = current_time
            self.fps_running = 0
        self.fps_running = self.fps_running + 1

    def _clean_up(self):
        """Cleans up resources"""
        try:
            if self.video_writer:
                self.video_writer.release()
            if self.video_stream:
                self.video_stream.release()
        except Exception:
            print('Error when shutting down')

    def _load_labels(self):
        """Loads labels for model"""
        self.labels = open(self.labels_path).read().strip().split('\n')

    def _load_model_to_memory(self):
        """Load the ML model into memory"""
        self.network = cv2.dnn.readNetFromDarknet(self.model_config_path, self.model_weights_path)
        self.layer_names = [self.network.getLayerNames()[i[0] - 1] for i in self.network.getUnconnectedOutLayers()]

    def _is_within_acceptable_confidence(self, confidence):
        """Determines if prediction is within acceptable confidence"""
        return confidence >= self.confidence_threshold

    def _apply_non_maxima_supression(self, box_list, confidence_list):
        """Applies non_maxima supression to results"""
        return cv2.dnn.NMSBoxes(
            box_list,
            confidence_list,
            self.confidence_threshold,
            self.non_maxima_supression_threshold
        )

    def _calculate_dimensions(self, detection, frame_height, frame_width):
        """Calculates dimensions around the current prediction"""
        box = detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
        (center_x, center_y, prediction_width, prediction_height) = box.astype("int")
        # get top left corner
        x = int(center_x - (prediction_width / 2))
        y = int(center_y - (prediction_height / 2))
        return [x, y, int(prediction_width), int(prediction_height)]

    def _set_fps_limit(self):
        """Sets a limit to the number of frames per second allowed"""
        fps_limit = os.environ.get('FPS_LIMIT', None)
        if fps_limit:
            fps_limit = int(fps_limit)
            if fps_limit < 1 or fps_limit > 120:
                raise Exception('Limit must be between 1 and 120')
            self.fps_wait_seconds = 1.0 / float(fps_limit)

    def process_frame(self, frame, draw_bounding_boxes=False):
        """Process and return a frame and results"""
        if self.should_display_fps:
            self._calculate_frames_per_second()
        if self.fps_wait_seconds:
            time.sleep(self.fps_wait_seconds)
        box_list = []
        confidence_list = []
        class_id_list = []
        results_list = []
        if type(frame) == type(None):
            return results_list
        frame_height, frame_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.network.setInput(blob)
        start_time = time.time()
        outputs = self.network.forward(self.layer_names)
        end_time = time.time()
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if self._is_within_acceptable_confidence(confidence):
                    box_list.append(self._calculate_dimensions(detection, frame_height, frame_width))
                    confidence_list.append(float(confidence))
                    class_id_list.append(class_id)
        frame_processing_time = end_time - start_time
        results = self._apply_non_maxima_supression(box_list, confidence_list)
        if len(results) > 0:
            for i in results.flatten():
                label = self.labels[class_id_list[i]]
                confidence = confidence_list[i]
                results_list.append(Detection(
                    frame=frame,
                    label=label,
                    confidence=confidence,
                    x=box_list[i][0],
                    y=box_list[i][1],
                    w=box_list[i][2],
                    h=box_list[i][3]
                ))
        return results_list
