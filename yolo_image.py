import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image


FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    yolo = YOLO()
    img_path = 'test_images/test00.jpg'
    try:
        image = Image.open(img_path)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.save('test_images/result_test00.jpg')
        r_image.show()
    yolo.close_session()

