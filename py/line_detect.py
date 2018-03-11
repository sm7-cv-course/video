import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

video_path = './../data/Видео0010.3gp'

cap = cv2.VideoCapture(video_path)

# while(cap.isOpened()):
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Canny Edge Detection:
    Threshold1 = 150;
    Threshold2 = 350;
    FilterSize = 5
    frame_edge = cv2.Canny(frame_gray, Threshold1, Threshold2, FilterSize)

    Rres = 1
    Thetares = 1 * np.pi / 180
    Threshold = 1
    minLineLength = 1
    maxLineGap = 100
    lines = cv2.HoughLinesP(frame_edge, rho = 1, theta = 1 * np.pi / 180,
                            threshold = 100, minLineLength = 100, maxLineGap = 50)
    # lines = cv2.HoughLinesP(frame_edge, Rres, Thetares, Threshold, minLineLength, maxLineGap)
    #if lines != None:
    if lines is not None:
        N = lines.shape[0]
        for i in range(N):
            x1 = lines[i][0][0]
            y1 = lines[i][0][1]
            x2 = lines[i][0][2]
            y2 = lines[i][0][3]
            cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    numpy_horizontal_concat1 = np.concatenate((frame, frame_hsv), axis=1)
    numpy_horizontal_concat2 = np.concatenate((frame_edge, frame_gray), axis=1)

    numpy_horizontal_concat_hsv1 = np.concatenate((frame_gray, frame_hsv[:,:,0]), axis=1)
    numpy_horizontal_concat_hsv2 = np.concatenate((frame_hsv[:,:,1], frame_hsv[:,:,2]), axis=1)
    concat_concat = np.concatenate((numpy_horizontal_concat_hsv1, numpy_horizontal_concat_hsv2), axis=0)

    cv2.imshow('frame color', numpy_horizontal_concat1)
    cv2.imshow('frame gray', numpy_horizontal_concat2)

    cv2.imshow('frame hsv', concat_concat)

    # plt.figure(), plt.imshow(frame), plt.title('Hough Lines'), plt.axis('off')
    # plt.show()

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
