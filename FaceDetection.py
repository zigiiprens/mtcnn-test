# This is a sample Python script that detect face in a image with MTCNN.
import cv2
import mtcnn

print(mtcnn.__version__)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def load_image(path):
    """Load image with OpenCV."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img


def detect_face(img):
    """Detect face with MTCNN Package."""
    detector = mtcnn.MTCNN()
    res = detector.detect_faces(img)
    return res


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

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    print_hi('MTCNN-TEST')
    image = load_image("data/face.jpg")
    results = detect_face(image)
    save_image(image, results, 'data/face.output.jpg')
    print(results)
