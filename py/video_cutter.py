import argparse
import cv2
import imutils
import numpy as np
import os
import sys


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--video", required=True, help="Path to source video.")
ap.add_argument("-o", "--out", required=False, help="Output folder path.")
ap.add_argument("-a", "--all", required=False, help="Save all channnels: i,r,g,b,h,s,v.", action='store_true')
ap.add_argument("-w", "--rgb", required=False, help="Save cut of original image(RGB).", action='store_true')
ap.add_argument("-i", "--gray", required=False, help="Save gray channnel(intensity).", action='store_true')
ap.add_argument("-r", "--red", required=False, help="Save red channnel.", action='store_true')
ap.add_argument("-g", "--green", required=False, help="Save green channnel.", action='store_true')
ap.add_argument("-b", "--blue", required=False, help="Save blue channnel.", action='store_true')
ap.add_argument("-u", "--hue", required=False, help="Save hue channnel.", action='store_true')
ap.add_argument("-t", "--sat", required=False, help="Save saturation channnel.", action='store_true')
ap.add_argument("-v", "--val", required=False, help="Save value channnel.", action='store_true')
ap.add_argument("-d", "--rotate", required=False, help="Rotate image.")
ap.add_argument("-c", "--cut", required=False, help="Save ROI.", action='store_true')
ap.add_argument("-W", "--width", required=False, help="width of ROI.")
ap.add_argument("-H", "--height", required=False, help="height of ROI.")
ap.add_argument("-x", "--X", required=False, help="left x coord of ROI.")
ap.add_argument("-y", "--Y", required=False, help="top y coord of ROI.")
ap.add_argument("-p", "--press", required=False, help="Save on 's' button press.", action='store_true')
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

if args["cut"]:
    if (args["width"] and args["height"] and args["X"] and args["Y"]) is None:
        print("You must specify width, height, x and y for cut!")
    else:
        x = int(args["X"])
        y = int(args["Y"])
        w = int(args["width"])
        h = int(args["height"])

out_dir = "."
if args["out"] is not None:
    out_dir = args["out"]

if args["rgb"] or args["all"]:
    os.makedirs(out_dir + "/rgb", mode=0o777, exist_ok=True)

if args["red"] or args["all"]:
    os.makedirs(out_dir + "/red", mode=0o777, exist_ok=True)

if args["green"] or args["all"]:
    os.makedirs(out_dir + "/green", mode=0o777, exist_ok=True)

if args["blue"] or args["all"]:
    os.makedirs(out_dir + "/blue", mode=0o777, exist_ok=True)

if args["hue"] or args["all"]:
    os.makedirs(out_dir + "/hue", mode=0o777, exist_ok=True)

if args["sat"] or args["all"]:
    os.makedirs(out_dir + "/sat", mode=0o777, exist_ok=True)

if args["val"] or args["all"]:
    os.makedirs(out_dir + "/val", mode=0o777, exist_ok=True)

if args["gray"] or args["all"]:
    os.makedirs(out_dir + "/gray", mode=0o777, exist_ok=True)

i=0
while(cap.isOpened()):
    i = i + 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    if args["rotate"] is not None:
        # Rotation angle in degree
        #frame = ndimage.rotate(frame, -float(args["rotate"]))
        frame = imutils.rotate(frame, angle=-float(args["rotate"]))

    if args["cut"]:
        frame = frame[y : y + h, x : x + w, :]

    # Our operations on the frame come here

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        #os._exit(0)
        break

    s_pressed = False
    if cv2.waitKey(25) & 0xFF == ord('s'):
        s_pressed = True

    if args["press"] == False or s_pressed:
        if args["gray"] or args["all"]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(out_dir + '/gray/' + 'gray_' + "{}".format(i) + '.png', gray)

        if args["sat"] or args["hue"] or args["val"] or args["all"]:
            frag_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if args["rgb"] or args["all"]:
            cv2.imwrite(out_dir + '/rgb/' + "{}".format(i) + '.png', frame)

        if args["red"] or args["all"]:
            cv2.imwrite(out_dir + '/red/' + 'red_' + "{}".format(i) + '.png', frame[:, :, 0])

        if args["green"] or args["all"]:
            cv2.imwrite(out_dir + '/green/' + 'green_' + "{}".format(i) + '.png', frame[:, :, 1])

        if args["blue"] or args["all"]:
            cv2.imwrite(out_dir + '/blue/' + 'blue_' + "{}".format(i) + '.png', frame[:, :, 2])

        if args["hue"] or args["all"]:
            cv2.imwrite(out_dir + '/hue/' + 'hue_' + "{}".format(i) + '.png', frag_hsv[:, :, 0])

        if args["sat"] or args["all"]:
            cv2.imwrite(out_dir + '/sat/' + 'sat_' + "{}".format(i) + '.png', frag_hsv[:, :, 1])

        if args["val"] or args["all"]:
            cv2.imwrite(out_dir + '/val/' + 'val_' + "{}".format(i) + '.png', frag_hsv[:, :, 2])

# Close output window
cv2.destroyAllWindows()

