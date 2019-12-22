from PIL import Image
from imageai.Detection import ObjectDetection
import os
import numpy as np


execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

class Makeup_artist(object):
    def __init__(self):
        pass

    def apply_makeup(self, img):
        print(type(img))
        num_array = np.asarray(img)
        print("shape -------- ", num_array.shape)
        result_img, detections = detector.detectObjectsFromImage(input_type="array", input_image=num_array, output_type="array", thread_safe=True)

        result_img = Image.fromarray(result_img)

        return result_img
