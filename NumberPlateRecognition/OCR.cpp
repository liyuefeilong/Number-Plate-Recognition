#include "OCR.h"

const char OCR::strCharacters[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z' };
const int OCR::numCharacters = 30;

CharSegment::CharSegment(){}
CharSegment::CharSegment(Mat i, Rect p)
{
	img = i;
	pos = p;
}

OCR::OCR()
{
	DEBUG = false;
	trained = false;
	saveSegments = false;
	charSize = 20;
}
OCR::OCR(string trainFile)
{
	DEBUG = false;
	trained = false;
	saveSegments = false;
	charSize = 20;

	// ��ȡOCR.xml�ļ�
	FileStorage fs;
	fs.open("OCR.xml", FileStorage::READ);
	Mat TrainingData;
	Mat Classes;
	fs["TrainingDataF15"] >> TrainingData;
	fs["classes"] >> Classes;

	train(TrainingData, Classes, 10);

}

Mat OCR::preprocessChar(Mat in){
	int h = in.rows;
	int w = in.cols;
	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = m / 2 - w / 2;
	transformMat.at<float>(1, 2) = m / 2 - h / 2;

	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

	Mat out;
	resize(warpImage, out, Size(charSize, charSize));

	return out;
}

bool OCR::verifySizes(Mat r){
	// ��ȷ�ĳ����ַ���߱�Ϊ45/77
	float aspect = 45.0f / 77.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.35;  // �������ﵽ35%
	float minHeight = 15;
	float maxHeight = 28;
	// ��С����
	float minAspect = 0.2;
	float maxAspect = aspect + aspect * error;
	// ��������
	float area = countNonZero(r);
	// bb����
	float bbArea = r.cols * r.rows;
	// ����ռ����İٷֱ�
	float percPixels = area / bbArea;

	// ��һ������ı��ʳ�����׼���ʵ�80%������Ϊ������Ϊ��ɫ�죬������һ���ַ�
	if (DEBUG)
		cout << "Aspect: " << aspect << " [" << minAspect << "," << maxAspect << "] " << "Area " << percPixels << " Char aspect " << charAspect << " Height char " << r.rows << "\n";
	if (percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect && r.rows >= minHeight && r.rows < maxHeight)
		return true;
	else
		return false;

}

// ��ֵ�ָ�
vector<CharSegment> OCR::segment(Plate plate){
	Mat input = plate.plateImg;
	vector<CharSegment> output;
	Mat thresholdImage;
	threshold(input, thresholdImage, 60, 255, CV_THRESH_BINARY_INV);
	if (DEBUG)
		imshow("Threshold plate", thresholdImage);
	Mat img_contours;
	thresholdImage.copyTo(img_contours);
	// �ҵ����ܵĳ��Ƶ�����
	vector< vector< Point> > contours;
	findContours(img_contours,
		contours, // �����������飬ÿһ��������һ��point���͵�vector��ʾ
		CV_RETR_EXTERNAL, // ��ʾֻ���������
		CV_CHAIN_APPROX_NONE); // �����Ľ��ư취������洢���е�������

	// �ڰ�ɫ��ͼ�ϻ�����ɫ������
	cv::Mat result;

	thresholdImage.copyTo(result);
	cvtColor(result, result, CV_GRAY2RGB);
	cv::drawContours(result, contours,
		-1,  // ���е�����������
		cv::Scalar(255, 0, 0), // ��ɫ
		1); // �ߴ�

	// ��ÿ������������ȡ��С������н��������
	vector<vector<Point> >::iterator itc = contours.begin();

	char res[20];
	int i = 0;
	// ��û�дﵽ�趨�Ŀ�߱�Ҫ����ȥ������
	while (itc != contours.end()) 
	{
		Rect mr = boundingRect(Mat(*itc));
		rectangle(result, mr, Scalar(0, 255, 0));
		// �ü�ͼ��
		Mat auxRoi(thresholdImage, mr);
		if (verifySizes(auxRoi)){
			auxRoi = preprocessChar(auxRoi);
			output.push_back(CharSegment(auxRoi, mr));
			//����ÿ���ַ�ͼƬ  
			sprintf(res, "PlateNumber%d.jpg", i);
			i++;
			imwrite(res, auxRoi);
			rectangle(result, mr, Scalar(0, 125, 255));
		}
		++itc;
	}
	if (DEBUG)
		cout << "Num chars: " << output.size() << "\n";

	if (DEBUG)
		imshow("SEgmented Chars", result);
	return output;
}

Mat OCR::ProjectedHistogram(Mat img, int t)
{
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j<sz; j++){
		Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = countNonZero(data);
	}

	// ��һ��ֱ��ͼ
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max>0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

Mat OCR::getVisualHistogram(Mat *hist, int type)
{

	int size = 100;
	Mat imHist;

	if (type == HORIZONTAL){
		imHist.create(Size(size, hist->cols), CV_8UC3);
	}
	else{
		imHist.create(Size(hist->cols, size), CV_8UC3);
	}

	imHist = Scalar(55, 55, 55);

	for (int i = 0; i<hist->cols; i++){
		float value = hist->at<float>(i);
		int maxval = (int)(value*size);

		Point pt1;
		Point pt2, pt3, pt4;

		if (type == HORIZONTAL)
		{
			pt1.x = pt3.x = 0;
			pt2.x = pt4.x = maxval;
			pt1.y = pt2.y = i;
			pt3.y = pt4.y = i + 1;

			line(imHist, pt1, pt2, CV_RGB(220, 220, 220), 1, 8, 0);
			line(imHist, pt3, pt4, CV_RGB(34, 34, 34), 1, 8, 0);

			pt3.y = pt4.y = i + 2;
			line(imHist, pt3, pt4, CV_RGB(44, 44, 44), 1, 8, 0);
			pt3.y = pt4.y = i + 3;
			line(imHist, pt3, pt4, CV_RGB(50, 50, 50), 1, 8, 0);
		}
		else
		{
			pt1.x = pt2.x = i;
			pt3.x = pt4.x = i + 1;
			pt1.y = pt3.y = 100;
			pt2.y = pt4.y = 100 - maxval;

			line(imHist, pt1, pt2, CV_RGB(220, 220, 220), 1, 8, 0);
			line(imHist, pt3, pt4, CV_RGB(34, 34, 34), 1, 8, 0);

			pt3.x = pt4.x = i + 2;
			line(imHist, pt3, pt4, CV_RGB(44, 44, 44), 1, 8, 0);
			pt3.x = pt4.x = i + 3;
			line(imHist, pt3, pt4, CV_RGB(50, 50, 50), 1, 8, 0);
		}
	}
	return imHist;
}

void OCR::drawVisualFeatures(Mat character, Mat hhist, Mat vhist, Mat lowData){
	Mat img(121, 121, CV_8UC3, Scalar(0, 0, 0));
	Mat ch;
	Mat ld;

	cvtColor(character, ch, CV_GRAY2RGB);

	resize(lowData, ld, Size(100, 100), 0, 0, INTER_NEAREST);
	cvtColor(ld, ld, CV_GRAY2RGB);

	Mat hh = getVisualHistogram(&hhist, HORIZONTAL);
	Mat hv = getVisualHistogram(&vhist, VERTICAL);

	Mat subImg = img(Rect(0, 101, 20, 20));
	ch.copyTo(subImg);

	subImg = img(Rect(21, 101, 100, 20));
	hh.copyTo(subImg);

	subImg = img(Rect(0, 0, 20, 100));
	hv.copyTo(subImg);

	subImg = img(Rect(21, 0, 100, 100));
	ld.copyTo(subImg);

	line(img, Point(0, 100), Point(121, 100), Scalar(0, 0, 255));
	line(img, Point(20, 0), Point(20, 121), Scalar(0, 0, 255));

	imshow("Visual Features", img);

	cvWaitKey(0);
}

// ������ȡ
Mat OCR::features(Mat in, int sizeData){
	//Histogram features
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);

	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));

	if (DEBUG)
		drawVisualFeatures(in, hhist, vhist, lowData);

	int numCols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);

	int j = 0;
	for (int i = 0; i<vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i<hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x<lowData.cols; x++)
	{
		for (int y = 0; y<lowData.rows; y++){
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}
	if (DEBUG)
		cout << out << "\n===========================================\n";
	return out;
}

// ���ڴ���������Ҫ�ľ�����ѵ�����ݡ����ǩ��
// �������ز����Ԫ������ѵ��һ��ʶ��ϵͳ
void OCR::train(Mat TrainData, Mat classes, int nlayers){
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = TrainData.cols;
	layers.at<int>(1) = nlayers;
	layers.at<int>(2) = numCharacters;
	ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	// ����һ���������д��n��ѵ�����ݣ��������Ϊm��
	Mat trainClasses;
	trainClasses.create(TrainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++)
	{
		for (int k = 0; k < trainClasses.cols; k++)
		{
			if (k == classes.at<int>(i))
				trainClasses.at<float>(i, k) = 1;
			else
				trainClasses.at<float>(i, k) = 0;
		}
	}
	Mat weights(1, TrainData.rows, CV_32FC1, Scalar::all(1));

	// ������ѧϰ
	ann.train(TrainData, trainClasses, weights);
	trained = true;
}

int OCR::classify(Mat f){
	int result = -1;
	Mat output(1, numCharacters, CV_32FC1);
	ann.predict(f, output);
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
	// ������Ҫ֪������������ֵ��x�������

	return maxLoc.x;
}

int OCR::classifyKnn(Mat f){
	int response = (int)knnClassifier.find_nearest(f, K);
	return response;
}
void OCR::trainKnn(Mat trainSamples, Mat trainClasses, int k){
	K = k;
	// ������ѧϰ
	knnClassifier.train(trainSamples, trainClasses, Mat(), false, K);
}

string OCR::run(Plate *input){

	// �ַ��ָ�
	vector<CharSegment> segments = segment(*input);

	for (int i = 0; i<segments.size(); i++){
		// ��ÿ���ַ�����Ԥ����ʹ�ö�����ͼ�������ͬ�Ĵ�С 
		Mat ch = preprocessChar(segments[i].img);
		if (saveSegments){
			stringstream ss(stringstream::in | stringstream::out);
			ss << "tmpChars/" << filename << "_" << i << ".jpg";
			imwrite(ss.str(), ch);
		}
		// Ϊÿ���ֶ���ȡ����
		Mat f = features(ch, 15);
		// ����ÿ�����ֽ��з���
		int character = classify(f);
		input->chars.push_back(strCharacters[character]);
		input->charsPos.push_back(segments[i].pos);
	}
	return "-";//input->str();
}
