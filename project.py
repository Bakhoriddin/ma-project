import cv2
import numpy as np

showCap=True #use a camera or not
cap=cv2.VideoCapture(0) #which camera to use, here it's set to main
classes=[] # we create classes for coco.names to process them further
whT=320 #We declare width and height
confThr=0.5 # Mininmum confidence
nmsThr=0.4  # Minimum confidence for the object in coco.names list

with open('coco.names','r') as f: #Here we start proccesing the coco.names file
    classes=f.read().rstrip('\n').split("\n") # We split and strip names in the file to pass them further

net=cv2.dnn.readNet('yolov3.cfg','yolov3.weights') # We declare a variable which will hold our YOLO library
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) # We declare that we want to use OpenCV as a back and for our Yolo library
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # We declare that we want to use CPU to process our program and images

def findObjects(outputs,img): # Function to proccess all the program
    wT,hT,_=img.shape # Creating width, height and declaring a channel
    bbox=[] # creating bounding box
    classIds=[] # Objects of coco.names
    confs=[] # Object for proccesing confidence
    for out in outputs: # Proccesing our output loop
        for det in out: # Main procces
            scores=det[5:] # Slicing the list
            classId=np.argmax(scores) # Getting data with max confidence
            confindence=scores[classId] #stating the data
            if confindence>confThr: 
                #mean-The Object is detected
                #process the objects as well as creating a bounding box
                w,h=int(det[2]*wT),int(det[3]*hT) # Width and height calculation
                x,y=int((det[0]*wT)-w/2),int((det[1]*hT)-h/2) # Calculation of the position of the bbox
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confindence))

    # Output proccessing
    indexes=cv2.dnn.NMSBoxes(bbox,confs,confThr,nmsThr)
    for i in range(len(bbox)):
        if i in indexes:
            x,y,w,h=bbox[i]
            label=str(classes[classIds[i]]) + ' ' + str(round(confs[0]*100, 2)) # Output text preparation
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3) # Displaying bounding box
            cv2.putText(img,label,(x,y+30),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3) # Displaying the text in our window

while True:
    if showCap: _,img=cap.read() # Checking for usage of cam
    else: img=cv2.imread('1.jpg') # If not process the image which is stored on device
    
    blob=cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False) #Creating and proccessing binary data in our case image
    net.setInput(blob) # Telling to procces this blob image
    layers=net.getLayerNames() # Getting layes
    outputN=[(layers[i-1]) for i in net.getUnconnectedOutLayers()] # Getting output names
    outputs=net.forward(outputN) # Getting the result of ourputs
    findObjects(outputs,img) # Passing data to our function

    cv2.imshow('win',img) # Creating a window to see the result
    cv2.waitKey(1) # Terminating the program