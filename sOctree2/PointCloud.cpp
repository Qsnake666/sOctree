#include "PointCloud.h"
#include "Utility.cuh"

//¶¨ÒåPoint×Ö¶Î
void PointCloud::Point::setPoint(double x, double y, double z, double a) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->a = a;
}

void PointCloud::Point::setPoint(double x, double y, double z) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->a = NO_ATTRIBUTE;
}

PointCloud::Point::Point(double x, double y, double z, double a) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->a = a;
}

PointCloud::Point::Point(double x, double y, double z) {
	this->x = x;
	this->y = y;
	this->z = z;
	this->a = NO_ATTRIBUTE;
}