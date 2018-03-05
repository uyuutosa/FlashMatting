#include "FlashMatting.h"


FlashMatting::FlashMatting(cv::Mat _I, cv::Mat _If):
	I(_I), If(_If), height(_I.rows), width(_I.cols)
{
	I.convertTo(I, CV_64FC3);
	If.convertTo(If, CV_64FC3);
}

FlashMatting::FlashMatting(std::string rawImagePath, std::string flashImagePath)
{
	cv::imread(rawImagePath).convertTo(I, CV_64FC3);
	cv::imread(flashImagePath).convertTo(If, CV_64FC3);
	height = I.rows;
	width  = I.cols;
}

void FlashMatting::build() {
	F     = cv::Mat::zeros(height, width, CV_64FC3);
	B     = F.clone();
	I_dot = If - I;
	F_dot = I_dot.clone();
	alpha = cv::Mat::zeros(height, width, CV_64FC1);

	F_mean     = cv::Mat(3, 1, CV_64FC1);     
	F_dot_mean = cv::Mat(3, 1, CV_64FC1); 
	B_mean     = cv::Mat(3, 1, CV_64FC1); 

	// separation of a image to two images(forweground and background).
	double F_mean_arr[3] = {};
	double B_mean_arr[3] = {};
	double F_dot_mean_arr[3] = {};

	for (int i = 0; i < height; i++) {
		unsigned char* Fp = F.ptr(i);
		unsigned char* I_dotp = I_dot.ptr(i);
		for (int j = 0; j < width*3; j=j+3) {
			Fp[j+0] = I_dotp[j+0] > 100 ? I_dotp[j+0]: 0;
			Fp[j+1] = I_dotp[j+2] > 100 ? I_dotp[j+1]: 0;
			Fp[j+2] = I_dotp[j+1] > 100 ? I_dotp[j+2]: 0;
		}
	}

	B = (I - F);

	for (int i = 0; i < height; i++) {
		unsigned char* Fp     = F.ptr(i);
		unsigned char* F_dotp = F_dot.ptr(i);
		unsigned char* Bp     = B.ptr(i);
		for (int j = 0; j < width*3; j=j+3) {
			F_mean_arr[0]     += Fp[j + 0];
			F_mean_arr[1]     += Fp[j + 1];
			F_mean_arr[2]     += Fp[j + 2];
			F_dot_mean_arr[0] += F_dotp[j + 0];
			F_dot_mean_arr[1] += F_dotp[j + 1];
			F_dot_mean_arr[2] += F_dotp[j + 2];
			B_mean_arr[0]     += Bp[j + 0];
			B_mean_arr[1]     += Bp[j + 1];
			B_mean_arr[2]     += Bp[j + 2];
		}
	}

	F_mean_arr[0]     /= height * width;
	F_mean_arr[1]     /= height * width;
	F_mean_arr[2]     /= height * width;
	F_dot_mean_arr[0] /= height * width;
	F_dot_mean_arr[1] /= height * width;
	F_dot_mean_arr[2] /= height * width;
	B_mean_arr[0]     /= height * width;
	B_mean_arr[1]     /= height * width;
	B_mean_arr[2]     /= height * width;

	F_mean.data     = (unsigned char*)&F_mean_arr;
	F_dot_mean.data = (unsigned char*)&F_dot_mean_arr;
	B_mean.data     = (unsigned char*)&B_mean_arr;

	O = cv::Mat::eye(3, 3, CV_64FC1);
	Z = cv::Mat::zeros(3, 3, CV_64FC1);

	A = cv::Mat(9, 9, CV_64FC1);
	b = cv::Mat(9, 1, CV_64FC1);

}

void FlashMatting::iterate(int itNum) {
	for (int it=0; it < itNum; it++)
	for (int i = 0; i < height; i++) {
		double* alphap = alpha.ptr<double>(i);
		double* Ip = I.ptr<double>(i);
		double* I_dotp = I_dot.ptr<double>(i);
		double* Fp = F.ptr<double>(i);
		double* Bp = B.ptr<double>(i);
		double* F_dotp = F_dot.ptr<double>(i);
		for (int j = 0; j < width*3; j=j+3) {
			alphap[j/3] = 
				(32 * 
				((Fp[j+0] - Bp[j+0]) * (Ip[j+0] - Bp[j+0]) + 
				 (Fp[j+1] - Bp[j+1]) * (Ip[j+1] - Bp[j+1]) + 
				 (Fp[j+2] - Bp[j+2]) * (Ip[j+2] - Bp[j+2])) + 32 *
				((F_dotp[j+0]) * (I_dotp[j+0]) + 
				 (F_dotp[j+1]) * (I_dotp[j+1]) + 
				 (F_dotp[j+2]) * (I_dotp[j+2]))) / 
			    (32 * 
				((Fp[j+0] - Bp[j+0]) * (Fp[j+0] - Bp[j+0]) + 
				 (Fp[j+1] - Bp[j+1]) * (Fp[j+1] - Bp[j+1]) + 
				 (Fp[j+2] - Bp[j+2]) * (Fp[j+2] - Bp[j+2])) + 32 *
				((F_dotp[j+0]) * (F_dotp[j+0]) + 
				 (F_dotp[j+1]) * (F_dotp[j+1]) + 
				 (F_dotp[j+2]) * (F_dotp[j+2])));  


			alpha_v = alphap[j / 3];
			cv::Mat F_sigma     = covMat((cv::Mat_<double>(3, 1) << Fp[0],     Fp[1],     Fp[2]),     F_mean);
			cv::Mat B_sigma     = covMat((cv::Mat_<double>(3, 1) << Bp[0],     Bp[1],     Bp[2]),     F_dot_mean);
			cv::Mat F_dot_sigma = covMat((cv::Mat_<double>(3, 1) << F_dotp[0], F_dotp[1], F_dotp[2]), B_mean);

			cv::Mat block_11 = F_sigma + alpha_v * alpha_v / 32;
			cv::Mat block_21 = O * alpha_v * (1 - alpha_v) * 32;
			cv::Mat block_31 = Z;
			cv::Mat block_12 = O * alpha_v * (1 - alpha_v) * 32;
			cv::Mat block_22 = B_sigma + alpha_v * alpha_v / 32;
			cv::Mat block_32 = Z;
			cv::Mat block_13 = Z;
			cv::Mat block_23 = Z;
			cv::Mat block_33 = F_dot_sigma + alpha_v * alpha_v / 32;
			cv::Mat block_b_1 = F_sigma * F_mean + Ip[j] * alpha_v / 32;
			cv::Mat block_b_2 = B_sigma * B_mean + Ip[j] * (1 - alpha_v) / 32;
			cv::Mat block_b_3 = F_dot_sigma * F_dot_mean + I_dotp[j] * alpha_v / 32;

			int shift_row(0), shift_col(0);
			A_arr[0+shift_row][0+shift_col] = block_11.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_11.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_11.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_11.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_11.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_11.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_11.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_11.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_11.ptr<double>(2)[2];

			shift_row = 3;
			shift_col = 0;
			A_arr[0+shift_row][0+shift_col] = block_21.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_21.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_21.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_21.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_21.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_21.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_21.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_21.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_21.ptr<double>(2)[2];

			shift_row = 6;
			shift_col = 0;
			A_arr[0+shift_row][0+shift_col] = block_31.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_31.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_31.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_31.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_31.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_31.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_31.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_31.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_31.ptr<double>(2)[2];

			shift_row = 0;
			shift_col = 3;
			A_arr[0+shift_row][0+shift_col] = block_12.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_12.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_12.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_12.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_12.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_12.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_12.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_12.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_12.ptr<double>(2)[2];

			shift_row = 3;
			shift_col = 3;
			A_arr[0+shift_row][0+shift_col] = block_22.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_22.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_22.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_22.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_22.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_22.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_22.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_22.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_22.ptr<double>(2)[2];

			shift_row = 6;
			shift_col = 3;
			A_arr[0+shift_row][0+shift_col] = block_32.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_32.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_32.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_32.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_32.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_32.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_32.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_32.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_32.ptr<double>(2)[2];

			shift_row = 0;
			shift_col = 6;
			A_arr[0+shift_row][0+shift_col] = block_13.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_13.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_13.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_13.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_13.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_13.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_13.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_13.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_13.ptr<double>(2)[2];

			shift_row = 3;
			shift_col = 6;
			A_arr[0+shift_row][0+shift_col] = block_23.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_23.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_23.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_23.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_23.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_23.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_23.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_23.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_23.ptr<double>(2)[2];

			shift_row = 6;
			shift_col = 6;
			A_arr[0+shift_row][0+shift_col] = block_33.ptr<double>(0)[0];
			A_arr[0+shift_row][1+shift_col] = block_33.ptr<double>(0)[1];
			A_arr[0+shift_row][2+shift_col] = block_33.ptr<double>(0)[2];
			A_arr[1+shift_row][0+shift_col] = block_33.ptr<double>(1)[0];
			A_arr[1+shift_row][1+shift_col] = block_33.ptr<double>(1)[1];
			A_arr[1+shift_row][2+shift_col] = block_33.ptr<double>(1)[2];
			A_arr[2+shift_row][0+shift_col] = block_33.ptr<double>(2)[0];
			A_arr[2+shift_row][1+shift_col] = block_33.ptr<double>(2)[1];
			A_arr[2+shift_row][2+shift_col] = block_33.ptr<double>(2)[2];

			b_arr[0][0] = block_b_1.ptr<double>(0)[0];
			b_arr[1][0] = block_b_1.ptr<double>(1)[0];
			b_arr[2][0] = block_b_1.ptr<double>(2)[0];
			b_arr[3][0] = block_b_2.ptr<double>(0)[0];
			b_arr[4][0] = block_b_2.ptr<double>(1)[0];
			b_arr[5][0] = block_b_2.ptr<double>(2)[0];
			b_arr[6][0] = block_b_3.ptr<double>(0)[0];
			b_arr[7][0] = block_b_3.ptr<double>(1)[0];
			b_arr[8][0] = block_b_3.ptr<double>(2)[0];

			A.data = (unsigned char*)&A_arr;
			b.data = (unsigned char*)&b_arr;

			cv::Mat out;

			cv::solve(A, b, out);

			Fp[j+0] = out.ptr<double>(0)[0];
			Fp[j+1] = out.ptr<double>(1)[0];
			Fp[j+2] = out.ptr<double>(2)[0];
			Bp[j+0] = out.ptr<double>(3)[0];
			Bp[j+1] = out.ptr<double>(4)[0];
			Bp[j+2] = out.ptr<double>(5)[0];
			F_dotp[j+0] = out.ptr<double>(6)[0];
			F_dotp[j+1] = out.ptr<double>(7)[0];
			F_dotp[j+2] = out.ptr<double>(8)[0];

		}

	}

}

void FlashMatting::solveAlpha() {

}

void FlashMatting::viewResult(){
    cv::Mat v[3];
	v[0] = alpha;
	v[1] = alpha;
	v[2] = alpha;
	cv::merge(v, 3, alpha);
	std::cout << alpha.channels();
	auto img = (imhan(I)->mult(imhan(alpha)))->convertTo(CV_8UC3);
	img->view();
	img->dump("./", "dump");

}

cv::Mat FlashMatting::covMat(cv::Mat m, cv::Mat mean){
	m = m - mean;
	return m * m.t();
}