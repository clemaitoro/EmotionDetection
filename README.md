# EmotionDetection
Small object detection into emotion recognition model trained with Yolov7 and tensorflow
The emotion detection is trained ond the FER-2013 dataset while the face recognition dataset is made by me by gathering lots of images from different places
I used this notebook found on kaggle to help me optimize my Tensorflow model https://www.kaggle.com/code/ammfat/facial-emotion-recognition-vgg16-fer2013

The way the model works is you first run the initial YOLO model using the Detect.py file and inputting the desired parameters python3 detect.py --weights best.pt --conf 0.25 --img-size 640 --source (path to images) --save-txt --nosave this detect the faces in the pictures and output labels of them 
After that you run the prediction.py where you input the path to your labeled and non labeled image and also the path for the model named "model.h5" After this you will have you prediction. This model runs the emotion categorization on each labeled image that the 1st model created and will output a full image of all the faces sorrounded by a square that specifies the emotion felt by that person. Ufortunately i cannot add the 2nd model since the file si too big for github
