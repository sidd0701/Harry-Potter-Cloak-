import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
time.sleep(3)

for i in range(30): 
    ret, background = cap.read()

background = np.flip(background, axis=1) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color range in HSV
    # Red has two ranges in HSV due to hue wrapping around 180
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine both red masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    
    # Create inverse mask for non-red areas
    inverse_mask = cv2.bitwise_not(red_mask)
    
    # Apply masks to get background where red object is detected
    res1 = cv2.bitwise_and(background, background, mask=red_mask)
    # Apply inverse mask to get current frame where no red object is detected
    res2 = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    
    # Combine both results to create the invisible cloak effect
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Red Invisible Cloak", final_output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
