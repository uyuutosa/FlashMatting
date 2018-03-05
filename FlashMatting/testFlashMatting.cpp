#include<iostream>
#include"FlashMatting.h"


int main() {


	FlashMatting obj("IMG_1098.JPG", "IMG_1099.JPG");
	obj.build();
	obj.iterate(1);
	obj.viewResult();

}