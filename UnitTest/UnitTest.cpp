#include<gtest\gtest.h>
#include<iostream>
#include"FlashMatting.h"

int main() {
	cv::Mat I = cv::imread("IMG_1098.JPG");
	cv::Mat If = cv::imread("IMG_1099.JPG");
	FlashMatting obj(I, If);
	obj.build();
	obj.iterate(1);
	obj.viewResult();
}

//TEST(FlashMattingTest, element){
//	FlashMatting obj("IMG_1098.JPG", "IMG_1099.JPG");
//	obj.build();
//	obj.iterate(1);
//	obj.viewResult();
//}