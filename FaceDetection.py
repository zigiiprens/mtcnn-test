# This is a sample Python script that detect face in a image with MTCNN.
import cv2
import time
import mtcnn
import argparse
import tensorflow as tf


def parser():
    parsing = argparse.ArgumentParser(description='Face Detection system with MTCNN.')

    parsing.add_argument('--image',
                         type=str,
                         default='data/face.jpg',
                         help='Input image filepath')
    return parsing.parse_args()


def load_image(path):
    """Load image with OpenCV."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def detect_face(img):
    """Detect face with MTCNN Package."""
    detector = mtcnn.MTCNN()
    start_time = time.time()
    res = detector.detect_faces(img)
    inf_time = (time.time() - start_time)
    return res, inf_time


def save_image(img, res, save_path):
    """Save image with BB and 5 Face Landmark."""
    # Result is an array with all the bounding boxes detected.
    bounding_box = res[0]['box']
    keypoints = res[0]['keypoints']

    cv2.rectangle(img,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (255, 155, 0),
                  10)

    cv2.circle(img, (keypoints['left_eye']), 2, (255, 155, 0), 10)
    cv2.circle(img, (keypoints['right_eye']), 2, (255, 155, 0), 10)
    cv2.circle(img, (keypoints['nose']), 2, (255, 155, 0), 10)
    cv2.circle(img, (keypoints['mouth_left']), 2, (255, 155, 0), 10)
    cv2.circle(img, (keypoints['mouth_right']), 2, (255, 155, 0), 10)

    return cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    print(f'MTCNN package version = {mtcnn.__version__}')
    print(f'OPENCV package version = {cv2.__version__}')
    print(f'TENSORFLOW package version = {tf.__version__}')
    print(f'TENSORFLOW GPU device = {tf.test.is_gpu_available()}')
    print(f'Hi, MTCNN-TEST')

    arguments = parser()

    image = load_image(arguments.image)
    results, inference_time = detect_face(image)
    if len(results) > 0:
        ret = save_image(image, results, 'data/face.output.jpg')
        if ret:
            print(f'Inference time = {inference_time}')
            print("-------------------------------------------------")
            for i in range(0, len(results)):
                print(f"Result No = {i}")
                print(f"Results BBOX = {results[i]['box']}")
                print(f"Results CONFIDENCE = {results[i]['confidence']}")
                print(f"Results KEYPOINTS = {results[i]['keypoints']}")
                print("-------------------------------------------------")
        else:
            print('Cannot save the processed image, checkout your folder path.')
    else:
        print('No detected faces in the given image.')
