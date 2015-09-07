#ifndef ImageEecognition_h
#define ImageEecognition_h

#include <string.h>
#include <vector>

#include "Plate.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>

using namespace std;
using namespace cv;

class ImageRecognition{
public:
	ImageRecognition();
	string filename;
	void setFilename(string f);
	bool saveRecognition;
	bool showSteps;
	vector<Plate> run(Mat input);

	vector<Plate> segment(Mat input);
	bool verifySizes(RotatedRect mr);
	Mat histeq(Mat in);
};

#endif
