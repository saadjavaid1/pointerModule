import cv2
import dlib
import numpy as np
import pyautogui as mouse





#get eye region coordinates 
def eye_on_mask(mask, side):
    points = [shape[i] for i in side]

    points = np.array(points, dtype=np.int32)

    mask = cv2.fillConvexPoly(mask, points, 255)

    return mask


    #get coordinates of the facial points
def shape_to_np(shape, dtype="int"):
	
	coords = np.zeros((68, 2), dtype=dtype)

	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


#extract pupil from eye region
def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)

        M = cv2.moments(cnt)

        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        #if right:
        cx += mid
        #cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        return np.asarray([cx, cy])
    except:
        pass



#start capturing video
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img = cv2.flip(img,1)
thresh = img.copy()

#recognize face in video
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('E:\\University\\FYP\\pythonProject\\shape_predictor_68_face_landmarks_GTX.dat')

#location  of eye regions
#left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]



#displaying image window
cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

#dispalying the threshold setter
def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 70, 255, nothing)


#reading the video input
while(True):
    ret, img = cap.read()
    img = cv2.flip(img,1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    #find faces in the video
    for rect in rects:
        #extract face
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        #extract eye region from detected face by 
        #drwaing the region  mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        #mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)

        # expand the white eye area
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

        #thresholding to extract eye balls from eye
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) 
        thresh = cv2.dilate(thresh, None, iterations=4) 
        thresh = cv2.medianBlur(thresh, 3)

        #invert the colors of background mask and eye mask
        thresh = cv2.bitwise_not(thresh)
        #leftEyePoint = contouring(thresh[:, 0:mid], mid, img)
        rightEyePoint = contouring(thresh[:, mid:], mid, img, True)

        
        
        #move mouse according to given points data

        try:
            mouse.moveTo(rightEyePoint[0],rightEyePoint[1])
        except:
            pass
        
        #print("left point", leftEyePoint,"right point", rightEyePoint)
        print("right point", rightEyePoint)
        
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == 27: #escape key
        break
    
cap.release()
cv2.destroyAllWindows()