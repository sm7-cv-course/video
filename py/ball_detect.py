import cv2
import numpy as np

cap = cv2.VideoCapture()

# while(cap.isOpened()):
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(frame, 100, 200)
    edges_hsv = cv2.Canny(img_hsv, 100, 200)
    edges_hue = cv2.Canny(img_hsv[:,:,0], 10, 100)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame hue', img_hsv[:,:,0])
    cv2.imshow('Canny', edges)
    cv2.imshow('Canny HSV', edges_hsv)
    cv2.imshow('Canny Hue', edges_hue)

    frame = cv2.medianBlur(frame, 5)
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1, 200,
                            param1=50, param2=20, minRadius=100, maxRadius=1000)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        #circles = np.round(circles[0, :]).astype("int")
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            cv2.circle(frame, (i[0], i[1]), 2, (0,0,255), 3)

    cv2.imshow('frame', frame)

    frame_yellow = cv2.inRange(frame, (40, 140, 140), (80, 180, 180))
    frame_yellow_hsv = cv2.inRange(img_hsv, (25, 120, 0), (70, 180, 255))

    # Find contours.
    image, contours, hierarchy = cv2.findContours(frame_yellow_hsv, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_NONE)

    # Get rid of too small objects.
    contours_thr = []
    area_thresh = 100
    for cnt in contours:
        if len(cnt) > 3:
            if cv2.contourArea(cnt) > area_thresh:
                contours_thr.append(cnt)

    cv2.drawContours(frame, contours_thr, -1, (0, 255, 0), 3)

    #frame_bw = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                                 cv2.THRESH_BINARY, 113, 1)

    #cv2.imshow('frame_bw', frame_bw)

    #h,w = frame.shape
    #print(frame[frame.shape[0] / 2, frame.shape[1] / 2, 0])
    r = frame[int(frame.shape[0] / 2), int(frame.shape[1] / 2), 0]
    g = frame[int(frame.shape[0] / 2), int(frame.shape[1] / 2), 1]
    b = frame[int(frame.shape[0] / 2), int(frame.shape[1] / 2), 2]
    hue = img_hsv[int(frame.shape[0] / 2), int(frame.shape[1] / 2), 0]
    sat = img_hsv[int(frame.shape[0] / 2), int(frame.shape[1] / 2), 1]
    print((r,g,b))
    print((hue, sat))
    cv2.imshow('frame_yellow', frame_yellow)
    cv2.imshow('frame_yellow_hsv', frame_yellow_hsv)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
