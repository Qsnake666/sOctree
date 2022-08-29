#pragma once
#include "PointCloud.h"
#include "Octree.cuh"
//实现Octree功能类，Octree为其父类

#define NO_ATTRIBUTE -1
#define MIX_ATTRIBUTE 0

class Utility :public Octree
{
public:
	//Brief：生成点云八叉树索引
	void genOctree(Octree* oct, std::vector<PointCloud::Point> pointCloud, double resolution);

	//Brief：生成多个点云八叉树索引
	void genMultiOctree(std::vector < Octree*>& octMul, std::vector<PointCloud::Point> pointCloud, double resolution);

	//Brief：自动计算level
	int getLevel(Octree::Octree_Struct octreeOG, double radius, double resolution);

	//Brief：点定位至某一深度八叉树结构
	Octree::Octree_Struct locatePoints(Octree::Octree_Struct octree, PointCloud::Point point, int level);

	//Brief：radius search点云模式，或者精密块体模式
	void radiusSearch(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	//Brief：radius search智能模式，或者块体模式
	void radiusSearchCube(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud);

	//Brief：radius search自由模式，或者块体模式
	void radiusSearchCubeByLevel(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, int inputLevel);
protected:
	//Brief：radius search Mode1a, in cube
	void radiusSearchMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	//Brief：radius search Mode1b, in cube，重载level=1模式
	void radiusSearchMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree, bool l1);

	//Brief：radius search Mode2a, out of cube
	void radiusSearchMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	//Brief：radius search Mode2b, 重载
	void radiusSearchMode2(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius);

	//Brief：radius search Mode2c, out of cube，重载level=1模式
	void radiusSearchMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree, bool l1);

	//Brief：获取点
	void getPointOctree(Octree::Octree_Struct octree, std::vector<Octree::Octree_Struct>& vOctree);

	//Brief：获取下一Level子节点
	std::vector<Octree::Octree_Struct> getLowerOctree(Octree::Octree_Struct octree, PointCloud::Point point, double radius);

	//Brief：快速模式获取Cube显示最多8x8x8x8x8个方块，重载
	void radiusSearchCubeMode2(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius, int locLevel);

	//Brief：快速模式获取Cube
	void radiusSearchCubeModeByLevel(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius, int locLevel, int inputLevel);

	//Brief：块体radius搜索 Mode1，in cube
	void radiusSearchCubeMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int locLevel, std::vector<PointCloud::Point>& pointCloud);

	//Brief：块体radius搜索 Mode2，out of cube
	void radiusSearchCubeMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud);

	//Brief：块体radius搜索ByLevel Mode1，in cube
	void radiusSearchCubeByLevelMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int locLevel, int inputLevel, std::vector<PointCloud::Point>& pointCloud);

	//Brief：块体radius搜索ByLevel Mode2，out of cube
	void radiusSearchCubeByLevelMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int inputLevel, std::vector<PointCloud::Point>& pointCloud);

};

