#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointCloud.h"
#include "Octree.cuh"

class Visualization {
public:
	Visualization() {}

	typedef struct {
		int width;
		int height;
		int stride;
		double* elements;
	} Matrix;
	//Brief：初始化Matrix
	void initMatrix(Visualization::Matrix* A, int row, int col);

	//Brief：vector转Matrix
	void vectorToMatrix(std::vector<PointCloud::Point> pointCloud, Matrix* A);

	//Brief：vector转Matrix块体中心补齐4方向
	void vectorToMatrixCube(std::vector<PointCloud::Point> pointCloud, Matrix* A, double length);

	//Brief：矩阵转置
	void matrixTranspose(Matrix A, Matrix B);

	//Brief：旋转矩阵生成
	void genRotateMatrix(Matrix A, char axis, double rad);

	//Brief：平移矩阵生成
	void genTranslationMatrix(Matrix A, double tx, double ty, double tz);

	//Brief：点云旋转
	Matrix ptcRotate(Matrix A, Matrix B);

	//Brief：点云投影

	//Brief：CUDA&OpenGL VBO

	//Brief：OpenGL

protected:
	//Brief：矩阵乘法
	void matrixMul(Matrix A, Matrix B, Matrix C);

	//Brief：矩阵加法
	void matrixAdd(Matrix A, Matrix B, Matrix C);
};
