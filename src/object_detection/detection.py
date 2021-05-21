import cv2
import numpy as np
import time

class Detection():
    """Class for representing a detection in a particular image"""

    def __init__(self, frame, label, confidence, x, y, w, h):
        self.frame = frame
        self.label = label
        self.confidence = confidence
        self.created_at = int(time.time())
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def get_original_frame(self):
        """Returns original frame"""
        return self.frame

    def get_frame_with_bounding_box(self, label=True):
        """Returns the frame with a bounding box"""
        frame = self.frame
        colour = (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        if label:
            text = "{}: {:.4f}".format(self.label, self.confidence)
            font_scale = 1.5
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, 3)
        return frame

    def identify(self):
        """Identify the detection"""
        print('{} detected, {:.4f} confident'.format(self.label, self.confidence))

    def to_dict(self):
        return {
            'label': self.label,
            'confidence': self.confidence,
            'created_at': self.created_at,
        }
    
    def is_person(self):
        return self.label == 'person'
