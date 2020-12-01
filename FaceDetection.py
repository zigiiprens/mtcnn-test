#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Imports
import cv2
import time
import mtcnn
import argparse
import tensorflow as tf


# Version controls
print(f"MTCNN package version = {mtcnn.__version__ == '0.1.0'}")
# noinspection PyUnresolvedReferences
print(f"OPENCV package version = {cv2.__version__ >= '4.4.0'}")
print(f"TENSORFLOW package version = {tf.__version__ >= '2.2.0'}")
print(f"TENSORFLOW GPU device = {tf.config.list_physical_devices()}")
print(f"Hi, MTCNN-TEST")


# Parser function
def parser():
    """
    Parse the script arguments and returns it.
    :return:
        arguments
    """
    parsing = argparse.ArgumentParser(description='Face Detection system with MTCNN.')

    parsing.add_argument('--image',
                         type=str,
                         default='data/input/face.jpg',
                         help='Input image filepath')
    return parsing.parse_args()


# FaceDetection class
class FaceDetection:
    def __init__(self, path):
        self.path = path
        self.detector = mtcnn.MTCNN()

        self.image = None
        self.face_results = None
        self.face_numbers = None

    def clear_data(self):
        """
        This function clear all data of the class: "path", "image", "face_results" and "face_numbers".
        """
        self.path = None
        self.image = None
        self.face_results = None
        self.face_numbers = None

    def load_image(self) -> float:
        """
        Load image with OpenCV.
        :return:
            load_time: The time for loading the frame with OPENCV.
        """
        start_time = time.time()
        self.image = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        load_time = (time.time() - start_time)
        return load_time

    def detect_face(self) -> float:
        """
        Detect face with MTCNN Package.
        :return:
            inf_time: The inference time for detection of all faces present in the frame.
        """
        start_time = time.time()
        self.face_results = self.detector.detect_faces(self.image)
        inf_time = (time.time() - start_time)
        self.face_numbers = len(self.face_results)
        return inf_time

    def save_image(self, save_path) -> bool:
        """
        Save image with BB and 5 Face Landmark.
        :arg:
            save_path: The path which will be saved the output image.
        :return:
            A boolean status of the cv2.imwrite function.
        """
        for elem in range(0, self.face_numbers):
            bounding_box = self.face_results[elem]['box']
            keypoints = self.face_results[elem]['keypoints']

            cv2.rectangle(self.image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (255, 155, 0),
                          1)

            cv2.circle(self.image, (keypoints['left_eye']), 2, (255, 155, 0), 2)
            cv2.circle(self.image, (keypoints['right_eye']), 2, (255, 155, 0), 2)
            cv2.circle(self.image, (keypoints['nose']), 2, (255, 155, 0), 2)
            cv2.circle(self.image, (keypoints['mouth_left']), 2, (255, 155, 0), 2)
            cv2.circle(self.image, (keypoints['mouth_right']), 2, (255, 155, 0), 2)

        return cv2.imwrite(save_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # Local test
    arguments = parser()
    face_detector = FaceDetection(arguments.image)
    loading_time = face_detector.load_image()
    inference_time = face_detector.detect_face()

    if face_detector.face_numbers > 0:
        ret = face_detector.save_image('data/output/face.output.jpg')
        if ret:
            print(f'Image loading time = {loading_time}')
            print(f'Inference time = {inference_time}')
            print(f'Image shape = {face_detector.image.shape}')
            print("-------------------------------------------------")
            for i in range(0, face_detector.face_numbers):
                print(f"Result No = {i}")
                print(f"Results BBOX = {face_detector.face_results[i]['box']}")
                print(f"Results CONFIDENCE = {face_detector.face_results[i]['confidence']}")
                print(f"Results KEYPOINTS = {face_detector.face_results[i]['keypoints']}")
                print("-------------------------------------------------")
        else:
            print('Cannot save the processed image, checkout your folder path.')
    else:
        print('No detected faces in the given image.')
