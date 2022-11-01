#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "PointCloud.h"

class Octree {
public:
	Octree() {}
	//~Octree() { destory(octree); }

	//Brief���ڵ����ݽṹ
	typedef struct Octree_Node {
		int count;  //�ýڵ�������������ĸ���
		std::vector<PointCloud::Point> points;	//������Ҷ�ӽڵ㱣�����
		Octree_Node* nodes[8];	//8���ӽڵ�
		Octree_Node* parent;	//���ڵ�

		int level;	//�ýڵ�˲����е����
		PointCloud::Point center;	//�˲�����������λ��
		double length;	//�˲���С���ӿ��

		//��ʼ���˲��������ڵ㣬���ӿ�ȣ���ȣ�
		void init(Octree_Node* parent, double length, int level);

		//�ͷŸýڵ�
		void destory();
	}Octree_Node, * Octree_Struct;

	//Brief�����Ա
	Octree_Struct octree;	//�˲���root�ڵ�
	double octLength;	//�˲���root�ڵ���������
	std::vector<PointCloud::Point> pointCloud;
	double max_x;
	double max_y;
	double max_z;
	double min_x;
	double min_y;
	double min_z;

	void setPoint(std::vector<PointCloud::Point>& points);

	void maxValue(std::vector<PointCloud::Point> pointCloud, double& overall_max_x, double& overall_max_y, double& overall_max_z);
	
	void minValue(std::vector<PointCloud::Point> pointCloud, double& overall_min_x, double& overall_min_y, double& overall_min_z);

	//Brief����ʼ���˲�������Χ�д�СpointCloud��
	void CreatOctreeByPointCloud();

	//Brief���ݹ������˲����սڵ�
	void creat(Octree_Struct octree);

	//Brief����ӵ��Ƶ��˲����� 
	void addPointCloud(std::vector<PointCloud::Point> pointCloud);

	void addNode(Octree_Struct octree, PointCloud::Point point);

	//Brief�������е㶼��ӵ��˲����󣬰ѿսڵ��ͷŵ�
	void addNodeEnd(Octree_Struct octree);

	//Brief�����ٰ˲���
	void destory(Octree_Struct octree);
};