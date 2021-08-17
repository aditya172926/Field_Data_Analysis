from imageai.Detection import ObjectDetection
from imageai.Prediction import ImagePrediction

detector = ObjectDetection()
model_path = "./models/resnet50_coco_best_v2.0.1.h5"
input_path = "./input/img45.jpg"
output_path = "./output/result45.jpg"

detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()

detection = detector.detectCustomObjectsFromImage(input_image=input_path, 
            output_image_path=output_path)

for eachItem in detection:
    print(eachItem["name"], " : ", eachItem["percentage_probability"])   
    print('\n') 