from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
  multiprocessing.freeze_support()

  # image = cv2.imread(r"D:\Data\PYTHON_code\Improved Lightweight YOLOv5 Using Attention Mechanism for Satellite Components Recognition\yolov5\image\803.jpg")
  # image = cv2.imread(r"D:\Data\PYTHON_code\Improved Lightweight YOLOv5 Using Attention Mechanism for Satellite Components Recognition\yolov5\724c2a0898074f4f98b543e88b76772f.jpeg")
  # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # cv2.imwrite(r"D:\Data\PYTHON_code\Improved Lightweight YOLOv5 Using Attention Mechanism for Satellite Components Recognition\yolov5\gray_image.jpeg", gray_image)
  model = YOLO()
  model.predict(r"D:\Data\PYTHON_code\Improved Lightweight YOLOv5 Using Attention Mechanism for Satellite Components Recognition\yolov5\灰度图\1.jpg",save = True,show_labels=True,line_width=1)

  # model.train(data='Satellite Components Recognition.yaml', epochs=100, batch = 32)
  
  # model.save