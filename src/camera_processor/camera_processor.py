import cv2
import threading

from object_detection.object_detector import ObjectDetector


detector = ObjectDetector()


class CameraBufferCleanerThread(threading.Thread):
    # https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()


class CameraProcessor():
    """Class for processing input from camera"""

    def __init__(self, stream_uri, name='Undefined Camera'):
        self.stream_uri = stream_uri
        self.name = name
        self.client = cv2.VideoCapture(stream_uri)
        self.client.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.thread = CameraBufferCleanerThread(self.client)

    def get_still(self):
        """Returns a still from the camera"""
        if self.thread.last_frame is not None:
            return self.thread.last_frame

    def get_detections(self):
        """Returns a list of detections visible in the camera"""
        frame = self.get_still()
        return detector.process_frame(frame, False)
