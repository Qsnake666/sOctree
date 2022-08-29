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
	//Brief����ʼ��Matrix
	void initMatrix(Visualization::Matrix* A, int row, int col);

	//Brief��vectorתMatrix
	void vectorToMatrix(std::vector<PointCloud::Point> pointCloud, Matrix* A);

	//Brief��vectorתMatrix�������Ĳ���4����
	void vectorToMatrixCube(std::vector<PointCloud::Point> pointCloud, Matrix* A, double length);

	//Brief������ת��
	void matrixTranspose(Matrix A, Matrix B);

	//Brief����ת��������
	void genRotateMatrix(Matrix A, char axis, double rad);

	//Brief��ƽ�ƾ�������
	void genTranslationMatrix(Matrix A, double tx, double ty, double tz);

	//Brief��������ת
	Matrix ptcRotate(Matrix A, Matrix B);

	//Brief������ͶӰ

	//Brief��CUDA&OpenGL VBO

	//Brief��OpenGL

protected:
	//Brief������˷�
	void matrixMul(Matrix A, Matrix B, Matrix C);

	//Brief������ӷ�
	void matrixAdd(Matrix A, Matrix B, Matrix C);
};
