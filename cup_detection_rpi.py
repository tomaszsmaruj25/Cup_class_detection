import h5py
import tensorflow
import cv2
#from tensorflow.keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0) # CAPTURE VIDEO FRAME
new_model = tensorflow.keras.models.load_model('cupModel_v1.h5') 


width = 1920
height = 1080
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print('W: ', width, ', H: ', height) 

font = cv2.FONT_HERSHEY_DUPLEX


while True:
    # capture video frame 
    ret,frame = cap.read()
    frame = cv2.flip(frame, -1) # rotate image 180 degrees
    
    # prepare frame to class detection
    cup_img = frame.copy()
    cup_img = cv2.resize(cup_img, (300, 300))
    cup_img = np.expand_dims(cup_img, axis=0)
    cup_scaled = cup_img / 255
    
    # check prediction
    pred1 = new_model.predict_classes(cup_scaled)
    if pred1 == 0:
        result = 'CURVED'
    elif pred1 == 1:
        result = 'OPEN LID'
    elif pred1 == 2:
        result = 'PROPER'
    elif pred1 == 3:
        result = 'UNCLEAN'
    
    # show result on text bar
    cup_detected = frame.copy()
    cup_detected = cv2.resize(cup_detected, (int(width/4), int(height/4)))
    cup_detection = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.rectangle(cup_detected, (0,0),(100,30),[255,255,255], -1)
    cv2.putText(cup_detected, result, (20, 20), font, 1, (0,0,0), 3, cv2.LINE_AA)

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     writer.write(frame)
    
    # show frame
    #scaled_frame = frame.copy()
    #scaled_frame = cv2.resize(scaled_frame, (int(width/4), int(height/4)))
    cv2.imshow('frame', cup_detected)
    # end condition - press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release and destroy camera windows
cap.release()
# writer.release()
cv2.destroyAllWindows()
