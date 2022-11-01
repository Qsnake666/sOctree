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

	//Brief：节点数据结构
	typedef struct Octree_Node {
		int count;  //该节点往下所包含点的个数
		std::vector<PointCloud::Point> points;	//用来在叶子节点保存点云
		Octree_Node* nodes[8];	//8个子节点
		Octree_Node* parent;	//父节点

		int level;	//该节点八叉树中的深度
		PointCloud::Point center;	//八叉树格子中心位置
		double length;	//八叉树小格子宽度

		//初始化八叉树（父节点，格子宽度，深度）
		void init(Octree_Node* parent, double length, int level);

		//释放该节点
		void destory();
	}Octree_Node, * Octree_Struct;

	//Brief：类成员
	Octree_Struct octree;	//八叉树root节点
	double octLength;	//八叉树root节点正方体宽度
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

	//Brief：初始化八叉树（包围盒大小pointCloud）
	void CreatOctreeByPointCloud();

	//Brief：递归生产八叉树空节点
	void creat(Octree_Struct octree);

	//Brief：添加点云到八叉树中 
	void addPointCloud(std::vector<PointCloud::Point> pointCloud);

	void addNode(Octree_Struct octree, PointCloud::Point point);

	//Brief：当所有点都添加到八叉树后，把空节点释放掉
	void addNodeEnd(Octree_Struct octree);

	//Brief：销毁八叉树
	void destory(Octree_Struct octree);
};