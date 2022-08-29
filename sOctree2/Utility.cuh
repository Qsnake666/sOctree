#pragma once
#include "PointCloud.h"
#include "Octree.cuh"
//ʵ��Octree�����࣬OctreeΪ�丸��

#define NO_ATTRIBUTE -1
#define MIX_ATTRIBUTE 0

class Utility :public Octree
{
public:
	//Brief�����ɵ��ư˲�������
	void genOctree(Octree* oct, std::vector<PointCloud::Point> pointCloud, double resolution);

	//Brief�����ɶ�����ư˲�������
	void genMultiOctree(std::vector < Octree*>& octMul, std::vector<PointCloud::Point> pointCloud, double resolution);

	//Brief���Զ�����level
	int getLevel(Octree::Octree_Struct octreeOG, double radius, double resolution);

	//Brief���㶨λ��ĳһ��Ȱ˲����ṹ
	Octree::Octree_Struct locatePoints(Octree::Octree_Struct octree, PointCloud::Point point, int level);

	//Brief��radius search����ģʽ�����߾��ܿ���ģʽ
	void radiusSearch(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	//Brief��radius search����ģʽ�����߿���ģʽ
	void radiusSearchCube(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud);

	//Brief��radius search����ģʽ�����߿���ģʽ
	void radiusSearchCubeByLevel(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, int inputLevel);
protected:
	//Brief��radius search Mode1a, in cube
	void radiusSearchMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	//Brief��radius search Mode1b, in cube������level=1ģʽ
	void radiusSearchMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree, bool l1);

	//Brief��radius search Mode2a, out of cube
	void radiusSearchMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	//Brief��radius search Mode2b, ����
	void radiusSearchMode2(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius);

	//Brief��radius search Mode2c, out of cube������level=1ģʽ
	void radiusSearchMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree, bool l1);

	//Brief����ȡ��
	void getPointOctree(Octree::Octree_Struct octree, std::vector<Octree::Octree_Struct>& vOctree);

	//Brief����ȡ��һLevel�ӽڵ�
	std::vector<Octree::Octree_Struct> getLowerOctree(Octree::Octree_Struct octree, PointCloud::Point point, double radius);

	//Brief������ģʽ��ȡCube��ʾ���8x8x8x8x8�����飬����
	void radiusSearchCubeMode2(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius, int locLevel);

	//Brief������ģʽ��ȡCube
	void radiusSearchCubeModeByLevel(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius, int locLevel, int inputLevel);

	//Brief������radius���� Mode1��in cube
	void radiusSearchCubeMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int locLevel, std::vector<PointCloud::Point>& pointCloud);

	//Brief������radius���� Mode2��out of cube
	void radiusSearchCubeMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud);

	//Brief������radius����ByLevel Mode1��in cube
	void radiusSearchCubeByLevelMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int locLevel, int inputLevel, std::vector<PointCloud::Point>& pointCloud);

	//Brief������radius����ByLevel Mode2��out of cube
	void radiusSearchCubeByLevelMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int inputLevel, std::vector<PointCloud::Point>& pointCloud);

};

