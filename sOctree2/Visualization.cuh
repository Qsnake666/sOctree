#pragma once
#pragma warning(push)
#pragma warning(disable:6385)
#pragma warning(disable:6386)
/*Source Code*/
#pragma   warning(pop) 
#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointCloud.h"
#include "Octree.cuh"
//gpu矩阵计算模块，当GPU内存溢出时无法完成

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

	//Brief：初始化Matrix，elements仅申请内存
	void initEmptyMatrix(Visualization::Matrix* A, int row, int col);

	//Brief：初始化Matrix，elements为ones矩阵
	void initOnesMatrix(Visualization::Matrix* A, int row, int col);

	//Brief：vector转Matrix
	void vectorToMatrix(std::vector<PointCloud::Point> pointCloud, Matrix* A);

	//Brief：vector转Matrix块体中心补齐4方向
	void vectorToMatrixCube(std::vector<PointCloud::Point> pointCloud, Matrix* A, double length);

	//Brief：矩阵乘法
	void matrixMul(Matrix A, Matrix B, Matrix C);

	//Brief：矩阵转置
	void matrixTranspose(Matrix A, Matrix B);

	//Brief：矩阵加法
	void matrixAdd(const Matrix A, const Matrix B, Matrix C);

	//Brief：矩阵加法，重载
	void matrixAdd(const Matrix A, double B, Matrix C);

	//Brief：矩阵减法
	void matrixSub(const Matrix A, const Matrix B, Matrix C);

	//Brief：矩阵减法，重载
	void matrixSub(const Matrix A, double B, Matrix C);

	//Brief：矩阵排序
	void matrixQSort(Matrix A, Matrix B, Matrix C);

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
};
