#pragma once

class PointCloud
{
public:
	struct Point {
		double x;
		double y;
		double z;
		double a;
		void setPoint(double x, double y, double z, double a);
		void setPoint(double x, double y, double z);
		Point() {}
		Point(double x, double y, double z, double a);
		Point(double x, double y, double z);
	};
};

