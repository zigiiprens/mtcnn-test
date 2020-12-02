#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Imports
import cv2
import caer
import time
import mtcnn
import argparse
import tensorflow as tf


# Version controls
print(f"MTCNN package version = {mtcnn.__version__ == '0.1.0'}")
# noinspection PyUnresolvedReferences
print(f"OPENCV package version = {cv2.__version__ >= '4.4.0'}")
print(f"CAER package version = {caer.__version__ >= '1.9.3'}")
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

    parsing.add_argument('--pack',
                         type=str,
                         choices=['opencv', 'caer'],
                         default='caer',
                         help='Frame process package')
    parsing.add_argument('--image',
                         type=str,
                         default='data/input/image.jpg',
                         help='Input image filepath')
    parsing.add_argument('--video',
                         type=str,
                         default='data/input/video.mp4',
                         help='Input video filepath')
    return parsing.parse_args()


class InFrame:
    def __init__(self, args):
        """
        InFrame load image and video from selected path using [opencv, caer] packages.
        :param:
            args: The arguments, need to be: pack, image and video.
        """
        self.packs = ['opencv', 'caer']
        self.pack = args.pack
        self.img_path = args.image
        self.vid_path = args.video

        self.image = None

    def load_image(self) -> float:
        """
        Load image from path.
        :return:
            load_time: The time for loading the frame with selected package.
        """
        if self.pack == self.packs[0]:
            start_time = time.time()
            self.image = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
            load_time = (time.time() - start_time)
            return load_time
        elif self.pack == self.packs[1]:
            start_time = time.time()
            self.image = caer.imread(image_path=self.img_path, channels=3, rgb=False)
            load_time = (time.time() - start_time)
            return load_time
        else:
            raise Exception(f"Unresolved pack name {self.pack}.")

    def save(self, face_numbers, face_results, save_path) -> bool:
        """
        Save image with BB and 5 Face Landmark.
        :arg:
            save_path: The path which will be saved the output image.
        :return:
            A boolean status of the cv2.imwrite function.
        """
        for elem in range(0, face_numbers):
            bounding_box = face_results[elem]['box']
            keypoints = face_results[elem]['keypoints']

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

        if self.pack == self.packs[0]:
            return cv2.imwrite(save_path, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
        elif self.pack == self.packs[1]:
            # TODO: Open Issue to CAER, does not return "imsave" status.
            caer.imsave(save_path, self.image)
            return True


# FaceDetection class
class FaceDetection:
    def __init__(self, args):
        """
        FaceDetection class help to detect faces present in the frame with MTCNN and save the output.
        :param:
            args: The arguments, need to be: pack, image and video.
        """
        self.iframe = InFrame(args)
        self.detector = mtcnn.MTCNN()

        self.face_results = None
        self.face_numbers = None

    def clear_data(self):
        """
        Clear all data of the class: "image", "face_results" and "face_numbers".
        """
        self.face_results = None
        self.face_numbers = None

    def load_image(self) -> float:
        """
        Load image with InFrame.
        :return:
            load_time: The time for loading the frame with InFrame.
        """
        return self.iframe.load_image()

    def detect_face(self) -> float:
        """
        Detect face with MTCNN Package.
        :return:
            inf_time: The inference time for detection of all faces present in the frame.
        """
        start_time = time.time()
        self.face_results = self.detector.detect_faces(self.iframe.image)
        inf_time = (time.time() - start_time)
        self.face_numbers = len(self.face_results)
        return inf_time

    def save(self, save_path) -> bool:
        """
        Save image with InFrame.
        :arg:
            save_path: The path which will be saved the output image.
        :return:
            A boolean status of the cv2.imwrite function.
        """
        return self.iframe.save(self.face_numbers, self.face_results, save_path)


if __name__ == '__main__':
    # Local test
    arguments = parser()
    face_detector = FaceDetection(arguments)

    loading_time = face_detector.load_image()
    inference_time = face_detector.detect_face()
    print(f'InFrame image loading time = {loading_time}')
    print(f'Inference time = {inference_time}')

    if face_detector.face_numbers > 0:
        ret = face_detector.save('data/output/face.output.jpg')
        if ret:
            print(f'Image shape = {face_detector.iframe.image.shape}')
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
