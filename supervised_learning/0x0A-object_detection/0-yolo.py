#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        model = K.models.load_model(model_path)
        self.model = model
        classes = [line.strip() for line in open(classes_path)]
        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
