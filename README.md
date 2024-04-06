# EmotionDetection
Small object detection into emotion recognition model trained with Yolov7 and tensorflow
The emotion detection is trained ond the FER-2013 dataset while the face recognition dataset is made by me by gathering lots of images from different places
I used this notebook found on kaggle to help me optimize my Tensorflow model https://www.kaggle.com/code/ammfat/facial-emotion-recognition-vgg16-fer2013

The way the model works is you first run the initial YOLO model using the Detect.py file and inputting the desired parameters python3 detect.py --weights best.pt --conf 0.25 --img-size 640 --source (path to images) --save-txt --nosave
After that you run the prediction.py where you input the path to your labeled and non labeled image and also the path for the model named "model.h5" After this you will have you prediction
