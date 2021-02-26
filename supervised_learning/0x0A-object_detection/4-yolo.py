#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K
import numpy as np
import cv2
import glob


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

    def sig(self, X):
        """sigmoid function"""
        return 1 / (1 + np.exp(-X))

    def process_outputs(self, outputs, image_size):
        """process output"""
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []
        for i, out in enumerate(outputs):
            grid_height, grid_width, anchor_boxes = out.shape[:-1]
            boxes.append(out[:, :, :, :4])
            box_confidences.append(self.sig(out[:, :, :, 4:5]))
            box_class_probs.append(self.sig(out[:, :, :, 5:]))
            t_x = boxes[i][:, :, :, 0]
            t_y = boxes[i][:, :, :, 1]
            t_w = boxes[i][:, :, :, 2]
            t_h = boxes[i][:, :, :, 3]
            cx = np.indices((grid_height, grid_width, anchor_boxes))[1]
            cy = np.indices((grid_height, grid_width, anchor_boxes))[0]
            bx = (self.sig(t_x) + cx) / grid_width
            by = (self.sig(t_y) + cy) / grid_height
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            input_width = self.model.input.shape[1].value
            input_height = self.model.input.shape[2].value
            bw = pw * np.exp(t_w) / input_width
            bh = ph * np.exp(t_h) / input_height
            x1 = bx - bw / 2
            x2 = x1 + bw
            y1 = by - bh / 2
            y2 = y1 + bh
            boxes[i][:, :, :, 0] = x1 * image_width
            boxes[i][:, :, :, 1] = y1 * image_height
            boxes[i][:, :, :, 2] = x2 * image_width
            boxes[i][:, :, :, 3] = y2 * image_height
        return boxes, box_confidences, box_class_probs

    @staticmethod
    def load_images(folder_path):
        """load images"""
        images = []
        image_paths = glob.glob(folder_path + '/*')
        for image in image_paths:
            images.append(cv2.imread(image, 3))
        return images, image_paths
