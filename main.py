import cv2 
import numpy as np


min_width = 80
min_height = 80

cap = cv2.VideoCapture('test.mp4')
clp = 550
total = 0
of = 6
#using substractor algorithm

a = cv2.bgsegm.createBackgroundSubtractorMOG() #import substractor algorithm from cv2

def center_point(x,y,w,h):
    x1 = int(w/2)
    y1 = int(w/2)
    cx = x+x1
    cy = y+y1
    return cx,cy

no_dectet =[ ]

while True:
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert into grey
    blur = cv2.GaussianBlur(grey,(3,3),5)

    img_sub = a.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE,kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE,kernel)
    counter,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #counter line 
    cv2.line(frame,(25,clp),(1200,clp),(0,255,0),3)

    for (i,channel) in enumerate(counter):
        (x,y,w,h) = cv2.boundingRect(channel)
        val_counter = (w>= min_width) and (h>=min_height)
        if not val_counter:
            continue
        cv2.rectangle(frame,(x,y),(x+w ,y+h),(0,0,255),2)
        cv2.putText(frame,"Vehicle Count:"+str(total),(x,y-20),cv2.FONT_HERSHEY_TRIPLEX,1,(255,277,0),1)
        
        c = center_point(x,y,w,h)
        no_dectet.append(c)
        #cv2.circle(frame,c,4,(0,0,255),-1)
        for (x,y) in no_dectet:
            if y<(clp+of) and y > (clp-of):
                total += 1
            cv2.line(frame,(25,clp),(1200,clp),(255,127,0),3)
            no_dectet.remove((x,y))
            print(str(total))
    
    cv2.putText(frame,"Vehicle Count:"+str(total),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)


 


    cv2.imshow('video...',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):#press q to exit
        break
cv2.destroyAllWindows()
cap.release()