#include "ImageRecognition.h"

void ImageRecognition::setFilename(string name) {
	filename = name;
}

ImageRecognition::ImageRecognition(){
	showSteps = false;
	saveRecognition = false;
}

bool ImageRecognition::verifySizes(RotatedRect ROI){
	// �������ó���Ĭ�ϲ���������ʶ������������Ƿ�ΪĿ�공��
	float error = 0.4;
	// ���������ƿ�߱�: 520 / 110 = 4.7272
	float aspect = 4.7272;
	// �趨�����������С/���ߴ磬���ڴ˷�Χ�ڵĲ�����Ϊ����
	int min = 15 * aspect * 15;    // 15������
	int max = 125 * aspect * 125;  // 125������
	float rmin = aspect - aspect*error;
	float rmax = aspect + aspect*error;

	int area = ROI.size.height * ROI.size.width;
	float r = (float)ROI.size.width / (float)ROI.size.height;
	if (r<1)
		r = (float)ROI.size.height / (float)ROI.size.width;

	// �ж��Ƿ�������ϲ���
	if ((area < min || area > max) || (r < rmin || r > rmax))
		return false;
	else
		return true;
}

// ��ͼ�����ֱ��ͼ���⴦����������
Mat ImageRecognition::histeq(Mat ima)
{
	Mat imt(ima.size(), ima.type());
	// ������ͼ��Ϊ��ɫ����Ҫ��HSV�ռ�����ֱ��ͼ���⴦��
	// ��ת����RGB��ʽ
	if (ima.channels() == 3)
	{
		Mat hsv;
		vector<Mat> hsvSplit;
		cvtColor(ima, hsv, CV_BGR2HSV);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, imt, CV_HSV2BGR);
	}
	// ������ͼ��Ϊ�Ҷ�ͼ��ֱ����ֱ��ͼ���⴦��
	else if (ima.channels() == 1){
		equalizeHist(ima, imt);
	}
	return imt;
}

// ͼ��ָ��
vector<Plate> ImageRecognition::segment(Mat input)
{
	vector<Plate> output;

	//nͼ��ת��Ϊ�Ҷ�ͼ
	Mat grayImage;
	cvtColor(input, grayImage, CV_BGR2GRAY);
	blur(grayImage, grayImage, Size(5, 5));  // ��ͼ������˲���ȥ������

	// ͨ������ӵ�������ı�Ե����������ʹ��sobel���Ӽ���Ե
	Mat sobelImage;
	Sobel(grayImage,       // ����ͼ��
		  sobelImage,	   // ���ͼ��
		  CV_8U,		   //���ͼ������
		  1,			   // x�����ϵĲ�ֽ���
		  0,			   // y�����ϵĲ�ֽ���
		  3,			   // ��չSobel�˵Ĵ�С��������1,3,5��7
		  1,			   // ���㵼��ֵʱ��ѡ���������ӣ�Ĭ��ֵ��1
		  0,			   // ��ʾ�ڽ������Ŀ��ͼ֮ǰ��ѡ��deltaֵ��Ĭ��ֵΪ0
		  BORDER_DEFAULT); // �߽�ģʽ��Ĭ��ֵΪBORDER_DEFAULT
	if (showSteps)
		imshow("Sobel", sobelImage);

	// ��ֵ�ָ�õ���ֵͼ�������õ���ֵ��Otsu�㷨�õ�
	Mat thresholdImage;
	// ����һ��8λͼ���Զ��õ��Ż�����ֵ
	threshold(sobelImage, thresholdImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	if (showSteps)
		imshow("Threshold Image", thresholdImage);

	// ��̬ѧ֮������
	// ����һ���ṹԪ��structuringElement��ά��Ϊ17*3
	Mat structuringElement = getStructuringElement(MORPH_RECT, Size(17, 3));
	// ʹ��morphologyEx�����õ��������Ƶ����򣨵����������ƺţ�
	morphologyEx(thresholdImage, thresholdImage, CV_MOP_CLOSE, structuringElement);
	if (showSteps)
		imshow("Close", thresholdImage);

	// �ҵ����ܵĳ��Ƶ�����
	vector< vector< Point> > contours;
	findContours(thresholdImage,
				 contours, // �����������飬ÿһ��������һ��point���͵�vector��ʾ
			     CV_RETR_EXTERNAL, // ��ʾֻ���������
				 CV_CHAIN_APPROX_NONE); // �����Ľ��ư취������洢���е�������

	// ��ÿ������������ȡ��С������н��������
	vector<vector<Point> >::iterator itc = contours.begin();
	vector<RotatedRect> rects;
	// ��û�дﵽ�趨�Ŀ�߱�Ҫ����ȥ������
	while (itc != contours.end()) 
	{
		RotatedRect ROI = minAreaRect(Mat(*itc));
		if (!verifySizes(ROI)){
			itc = contours.erase(itc);
		}
		else{
			++itc;
			rects.push_back(ROI);
		}
	}

	// �ڰ�ɫ��ͼ�ϻ�����ɫ������
	cv::Mat result;
	input.copyTo(result);
	cv::drawContours(result,
					 contours,
					 -1,				    // ���е�����������
					 cv::Scalar(255, 0, 0), // ��ɫ
					 1);					// �ߴ�

	// ʹ����ˮ����㷨�ü����ƻ�ȡ������������
	for (int i = 0; i< rects.size(); i++){

		circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
		// �õ���Ⱥ͸߶��н�С��ֵ���õ����Ƶ���С�ߴ�
		float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
		minSize = minSize - minSize * 0.5;
		// �ڿ����ĸ����������ɸ��������
		srand(time(NULL));
		// ��ʼ����ˮ����㷨�Ĳ���
		Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		// loDiff��ʾ��ǰ�۲�����ֵ���䲿����������ֵ���ߴ�����
		// �ò�������������֮������Ȼ���ɫ֮��������ֵ
		int loDiff = 30;
		// upDiff��ʾ��ǰ�۲�����ֵ���䲿����������ֵ���ߴ�����
		// �ò�������������֮������Ȼ���ɫ֮��������ֵ
		int upDiff = 30;
		int connectivity = 4; // ���ڿ����㷨����ͨ�ԣ���ȡ4����8
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		// ������־����Ϊ��������
		int flags = connectivity + // ���ڿ����㷨����ͨ�ԣ���ȡ4����8
				    (newMaskVal << 8) +
					CV_FLOODFILL_FIXED_RANGE + // ���øñ�ʶ�����ῼ�ǵ�ǰ��������������֮��Ĳ�
					CV_FLOODFILL_MASK_ONLY; // ��������ȥ���ı�ԭʼͼ��, ����ȥ�����ģͼ��
		for (int j = 0; j < NumSeeds; j++){
			Point seed;
			seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
			circle(result, seed, 1, Scalar(0, 255, 255), -1);
			// ��������㷨������������
			int area = floodFill(input,
								 mask,
								 seed,
								 Scalar(255, 0, 0),
								 &ccomp,
								 Scalar(loDiff, loDiff, loDiff),
								 Scalar(upDiff, upDiff, upDiff),
								 flags);
		}
		if (showSteps)
			imshow("MASK", mask);

		// �õ��ü�����󣬼������Ч�ߴ�
		// ����ÿ������İ�ɫ���أ��ȵõ���λ��
		// ��ʹ��minAreaRect������ȡ��ӽ��Ĳü�����
		vector<Point> pointsInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for (; itMask != end; ++itMask)
			if (*itMask == 255)
				pointsInterest.push_back(itMask.pos());

		RotatedRect minRect = minAreaRect(pointsInterest);

		if (verifySizes(minRect)){
			// ��ת����ͼ
			Point2f rect_points[4]; minRect.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);

			// �õ���תͼ������ľ���
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			if (r<1)
				angle = 90 + angle;
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);

			// ͨ������任��ת�����ͼ��
			Mat img_rotated;
			warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);

			// ���ü�ͼ��
			Size rect_size = minRect.size;
			if (r < 1)
				swap(rect_size.width, rect_size.height);
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			Mat resultResized;
			resultResized.create(33, 144, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
			// Ϊ����������Ӱ�죬�Բü�ͼ��ʹ��ֱ��ͼ���⻯����
			Mat grayResult;
			cvtColor(resultResized, grayResult, CV_BGR2GRAY);
			blur(grayResult, grayResult, Size(3, 3));
			grayResult = histeq(grayResult);
			if (saveRecognition){
				stringstream ss(stringstream::in | stringstream::out);
				ss << "tmp/" << filename << "_" << i << ".jpg";
				imwrite(ss.str(), grayResult);
			}
			output.push_back(Plate(grayResult, minRect.boundingRect()));
		}
	}
	if (showSteps)
		imshow("Contours", result);

	return output;
}

vector<Plate> ImageRecognition::run(Mat input)
{
	vector<Plate> tmp = segment(input);
	// ���ؼ����
	return tmp;
}