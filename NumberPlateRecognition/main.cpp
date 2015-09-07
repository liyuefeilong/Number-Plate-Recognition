#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv/ml.h>

#include <iostream>
#include <vector>

#include "ImageRecognition.h"
#include "OCR.h"

using namespace std;
using namespace cv;

string getFilename(string s) {

	char sep = '/';
	char sepExt = '.';

#ifdef _WIN32
	sep = '\\';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != string::npos) {
		string fn = (s.substr(i + 1, s.length() - i));
		size_t j = fn.rfind(sepExt, fn.length());
		if (i != string::npos) {
			return fn.substr(0, j);
		}
		else{
			return fn;
		}
	}
	else{
		return "";
	}
}

int main(int argc, char** argv)
{
	cout << "OpenCV Automatic Number Plate Recognition\n";
	char* filename;
	Mat InputImage;

	// ��ʾ�Ƿ���ȷ�������ͼ��
	if (argc >= 2)
	{
		filename = argv[1];
		InputImage = imread(filename, 1); // ����ͼ���Զ�ת��Ϊ�Ҷ�ͼ��
	}
	else{
		printf("Use:\n\t%s image\n", argv[0]);
		return 0;
	}

	string filename_whithoutExt = getFilename(filename);
	cout << "working with file: " << filename_whithoutExt << "\n";

	// ���Ƽ�ⲿ��
	ImageRecognition detector;
	detector.setFilename(filename_whithoutExt);
	detector.saveRecognition = false;
	detector.showSteps = false;
	vector<Plate> PosibleRecognition = detector.run(InputImage);

	// ѵ��SVM,����ѵ���Ͳ��Ե�ͼ�����ݱ�����SVM.xml�ļ���
	FileStorage fs;
	fs.open("SVM.xml", FileStorage::READ);
	Mat SVM_TrainingData;
	Mat SVM_Classes;
	fs["TrainingData"] >> SVM_TrainingData;
	fs["classes"] >> SVM_Classes;
	// ����SVM�Ļ�������
	CvSVMParams SVM_params;  // CvSVMParams�ṹ���ڶ����������
	SVM_params.svm_type = CvSVM::C_SVC;		// SVM����
	SVM_params.kernel_type = CvSVM::LINEAR; // ����ӳ��
	SVM_params.degree = 0;
	SVM_params.gamma = 1;
	SVM_params.coef0 = 0;
	SVM_params.C = 1;
	SVM_params.nu = 0;
	SVM_params.p = 0;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);

	// ������ѵ��������
	CvSVM svmClassifier(SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params);

	// ʹ��ѵ���õķ������Բü�ͼ����з��࣬�жϸ������Ƿ�Ϊ����
	vector<Plate> plates;
	for (int i = 0; i< PosibleRecognition.size(); i++)
	{
		Mat img = PosibleRecognition[i].plateImg; 
		Mat p = img.reshape(1, 1);
		// ��ͼ��ת��Ϊ����ֵ����Ϊ�����͵ĵ�ͨ��ͼ��
		p.convertTo(p, CV_32FC1); 

		int response = (int)svmClassifier.predict(p); // ����SVM������
		if (response == 1)
			plates.push_back(PosibleRecognition[i]);
	}

	// �����������ж�Ϊ����ʱ���1������Ϊ0
	cout << "Num plates detected: " << plates.size() << "\n";

	// �Ա��궨Ϊ���Ƶ��������OCR�ָ�
	OCR ocr("OCR.xml");
	ocr.saveSegments = true;
	ocr.DEBUG = false;
	ocr.filename = filename_whithoutExt;
	for (int i = 0; i< plates.size(); i++){
		Plate plate = plates[i];

		string plateNumber = ocr.run(&plate);
		string licensePlate = plate.str();
		cout << "================================================\n";
		cout << "License plate number: " << licensePlate << "\n";
		cout << "================================================\n";
		rectangle(InputImage, plate.position, Scalar(0, 0, 200));
		putText(InputImage, licensePlate, Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 200), 2);
		if (false){
			imshow("Plate Detected seg", plate.plateImg);
			cvWaitKey(0);
		}

	}
	imshow("Numbers of the Plate", InputImage);
	for (;;)
	{
		int c;
		c = cvWaitKey(10);
		if ((char)c == 27)
			break;
	}
	return 0;
}