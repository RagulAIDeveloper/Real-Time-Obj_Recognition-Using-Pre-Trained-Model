# import the necessary packages
import numpy
import imutils
import cv2

#initialization caffemodel
prototxt = "MobileNetSSD.txt";
model = "MobileNetSSD_deploy.caffemodel";
confThresh = 0.6;

CLASSES = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmoniter","mobile"]

COLORS = numpy.random.uniform(0,255,size=(len(CLASSES),3))#random color

#loading model
model_loaded = cv2.dnn.readNetFromCaffe(prototxt,model)
print("model loaded....!")

#camera id initialization
cam = cv2.VideoCapture(0);

while True:
    _,frame = cam.read(); # camera reading
    frame =imutils.resize(frame,width=1000); #frame size
    #getting the width and height of frameimg
    (h,w) =frame.shape[:2]
    #resize the img
    imResizeBlob = cv2.resize(frame,(300,300))
    # convert the blob img
    blob = cv2.dnn.blobFromImage(imResizeBlob,0.007843,(300,300),127.5)
    #setinput
    model_loaded.setInput(blob);
    #forward
    detections = model_loaded.forward()
    detshape = detections.shape[2]
    #print('shape',detshape)
    for i in numpy.arange(0,detshape):
        confidence = detections[0,0,i,2]
        print(confidence)
        if confidence>confThresh:
            #getting the id
            idx = int(detections[0,0,i,1])
            print('id', idx)
            print("ClassID :",detections[0,0,i,1])
            #getting the coordinates
            box = detections[0,0,i,3:7] * numpy.array([w,h,w,h])
            (startX,startY,endX,endY) = box.astype("int")
            label = "{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            #draw rectangle
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            print('k',COLORS)
            if startY-15>15:
                y = startY-15
            else:
                startY + 15
        cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
