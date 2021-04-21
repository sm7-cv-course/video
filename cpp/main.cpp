#include <cstdio>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

static const unsigned int low_h = 25;
static const unsigned int low_s = 120;
static const unsigned int low_v = 0;
static const unsigned int high_h = 70;
static const unsigned int high_s = 180;
static const unsigned int high_v = 255;

typedef std::vector<std::vector<cv::Point> > Contours;

cv::Mat
get_layer(cv::Mat img, int i_layer) {

	if(i_layer > img.channels()) {
		return cv::Mat();
	}

    cv::Mat layer = cv::Mat(cv::Size(img.rows, img.cols), CV_8UC1);

    for(int i=0; i < img.rows; ++i) {
        for(int j=0; j < img.cols; ++j) {
            layer.at<unsigned char>(i,j) = img.at<cv::Vec3b>(i,j)[i_layer];
        }
    }

    return layer;
}

void
filter_bw(cv::Mat src, cv::Mat dst) {
	cv::erode(src, dst, cv::Mat());
}

/**
 * Seeks circular blobs in img among contours.
 *
 * Returns point
 */
cv::Point2f
find_circle(cv::Mat img, Contours contours) {
	cv::Point2f out_center;

	double min_area_dist = img.cols * img.rows;
	int min_index = 0;

	for(size_t i=0; i < contours.size(); ++i) {
		double area = cv::contourArea(contours[i], false);
		float radius;
		cv::Point2f center;
		cv::minEnclosingCircle(contours[i], center, radius);
		double area_dist = fabs(area - M_PI * radius * radius);
		if(min_area_dist > area_dist) {
			min_area_dist = area_dist;
			out_center = center;
			min_index = i;
		}
	}

	cv::drawContours(img, contours, min_index, cv::Scalar(255, 0, 0));

	return out_center;
}

int
ball_tracker() {
    cv::VideoCapture cap(0);

    if(!cap.isOpened())  // check if we succeeded
        return -1;

    cv::Mat frame, frame_gray, frame_hsv;
    cv::namedWindow("edges", 1);
    cv::namedWindow("Hue", 1);
    cv::namedWindow("frame_yellow", 1);


    for(;;)
    {
        cap >> frame; // get a new frame from camera
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame, frame_hsv, cv::COLOR_BGR2HSV);
        cv::GaussianBlur(frame_gray, frame_gray, cv::Size(7,7), 1.5, 1.5);
        cv::Mat frame_hue = get_layer(frame_hsv, 0);

        cv::Mat frame_yellow;
        cv::inRange(frame_hue, cv::Scalar(low_h,low_s,low_v), cv::Scalar(high_h,high_s,high_v), frame_yellow);

        cv::Mat frame_yellow_small;
        cv::resize(frame_yellow, frame_yellow_small, cv::Size(0,0), 0.125, 0.125);

        filter_bw(frame_yellow, frame_yellow);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(frame_yellow_small.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        cv::Point2f center = find_circle(frame_yellow_small, contours);

        cv::imshow("Hue", frame_hsv);
        cv::imshow("edges", frame_gray);
        cv::imshow("frame_yellow", frame_yellow_small);

        if(cv::waitKey(30) >= 0) break;
    }

    return 0;
}

int
main(int argc, char **argv)
{
    ball_tracker();

    return 0;
}


/* # while(cap.isOpened()):
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
        for i in circles[0,:]:
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
        break */
