import cv2 as cv

# Importing Image.
# Place path of image you want to detect.
img = cv.imread('group.jpg')

# Image Resize Function.
def rescaleFrame(frame,scale=0.2):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions =(width,height)

    return cv.resize(frame,dimensions,interpolation = cv.INTER_AREA)

# Call for resizing Image.
frame_resized_img = rescaleFrame(img)

# Converting image into greyscale
gray = cv.cvtColor(frame_resized_img, cv.COLOR_BGR2GRAY)

# Importing .xml files for Face and Mouth detection.
haar_cascade = cv.CascadeClassifier('haar_mouth.xml')
haar_cascade_face = cv.CascadeClassifier('haar_face.xml')
    
# Using xml files for detecting Face and mouth and storing in faces_react and mouth_react
faces_rect = haar_cascade_face.detectMultiScale(frame_resized_img, scaleFactor=1.1, minNeighbors=20)
mouth_rect = haar_cascade.detectMultiScale(frame_resized_img, scaleFactor=1.1, minNeighbors=20)
    

print(f'Number of faces found = {len(faces_rect)}')
print(f'Number of mouth found = {len(mouth_rect)}')


# Function to check weater given face has any mouth in lower half of face.
def ismouth(x,y,w,h) :
    for (xm,ym,wm,hm) in mouth_rect:
        half_faceY = int(y+(h/2))
        if (x < xm) and (half_faceY < ym) and (x+w > xm + wm) and (y+h > ym + hm) :
            cv.rectangle(frame_resized_img, (xm,ym), (xm+wm,ym+hm), (0,0,255), thickness=2)
            cv.putText(frame_resized_img,"\U0001f600",(xm+int(wm/2),int(ym+hm+(hm/5))),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,225),thickness=3)
            return 1
    return 0

ct=0      # Number of people who do not have face mask. 
found = 0 # Number of people who have face mask.


for (x,y,w,h) in faces_rect: # Looping over all the faces colected.
    # cv.rectangle(frame_resized_img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

    # calling ismouth function to check weather any mouth is present in lower half of current face.
    # If none no mouth found in current face that means current person have face mask and vice-a-versa.
    ans = ismouth(x,y,w,h)
    ct+=ans

    if ans==0: 
        found+=1
        cv.putText(frame_resized_img,"[*]",(x+int(w/2),int(y+h+(h/5))),cv.FONT_HERSHEY_COMPLEX,1.0,(0,225,0),thickness=3)

        
    
print(f'Mask Not found in {ct} people.')
print(f'Mask found in {found} people.')

# Showing image in a window.
cv.imshow('Detected mouth & face',frame_resized_img)

# To hold window untill 'd' key is pressed.
cv.waitKey(0)