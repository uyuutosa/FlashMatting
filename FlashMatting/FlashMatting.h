#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <imhan.h>


class FlashMatting {
public:

	FlashMatting(cv::Mat _I, cv::Mat _If);

	FlashMatting(std::string rawImagePath, std::string flashImagePath);

	void build();

	void iterate(int itNum);

	//void generateTrimap();

	//void calculateNonNormalizeCov();

	//void CollectSampleSet();

	//void InitializeAlpha();

	void solveAlpha();

	//void SolveBFF_();

	void viewResult();

	cv::Mat covMat(cv::Mat m, cv::Mat mean);

	cv::Mat I, If, F, B, I_dot, F_dot, alpha;
	
	cv::Mat F_mean;
	cv::Mat F_dot_mean;
	cv::Mat B_mean;

	double A_arr[9][9], b_arr[9][1];
	cv::Mat A, b;
	cv::Mat O, Z;
	double alpha_v;

	int height, width;
};