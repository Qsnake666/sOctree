#pragma once

class PointCloud
{
public:
	struct Point {
		double x;
		double y;
		double z;
		int a;
		void setPoint(double x, double y, double z, int a);
		void setPoint(double x, double y, double z);
		Point() {}
		Point(double x, double y, double z, int a);
		Point(double x, double y, double z);
	};
};

