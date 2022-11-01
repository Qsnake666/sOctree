#pragma once
#pragma warning(push)
#pragma warning(disable:6385)
#pragma warning(disable:6386)
/*Source Code*/
#pragma   warning(pop) 
#include "PointCloud.h"
#include "Octree.cuh"
#include <Eigen/Dense>
#include <cmath>
//实现Octree功能类，Octree为其父类

#define NO_ATTRIBUTE -1
#define MIX_ATTRIBUTE 0

class Utility :public Octree
{
public:
	/**
	 * @brief 生成点云八叉树索引
	 * @param *oct Octree类的实例
	 * @param pointCloud 模板为PointCloud::Point的点云向量
	 * @param resolution 生成八叉树索引最低一级的叶子节点的尺寸
	 * @return void
	 */
	void genOctree(Octree* oct, std::vector<PointCloud::Point> pointCloud, double resolution);

	/**
	 * @brief 生成多个点云八叉树索引
	 * @param &octMul 模板为Octree类的实例的向量
	 * @param pointCloud 模板为PointCloud::Point的点云向量
	 * @param resolution 生成八叉树索引最低一级的叶子节点的尺寸
	 * @return void
	 */
	void genMultiOctree(std::vector < Octree*>& octMul, std::vector<PointCloud::Point> pointCloud, double resolution);

	/**
	 * @brief 半径搜索时根据半径自动计算对应八叉树搜寻level
	 * @param octreeOG Octree类的根节点八叉树结构体指针
	 * @param radius 半径搜索半径
	 * @param resolution 生成八叉树索引最低一级的叶子节点的尺寸
	 * @return int 八叉树的层级
	 */
	int getLevel(Octree::Octree_Struct octreeOG, double radius, double resolution);

	/**
	 * @brief 给定点定位至规定深度八叉树结构
	 * @param octree Octree类的八叉树结构体指针
	 * @param point 给定三维点PointCloud::Point
	 * @param level 规定八叉树层级
	 * @return Octree::Octree_Struct Octree类的八叉树结构体指针
	 */
	Octree::Octree_Struct locatePoints(Octree::Octree_Struct octree, PointCloud::Point point, int level);

	/**
	 * @brief radius search点云模式，或者精密块体模式
	 * @param octreeOG Octree类的根节点八叉树结构体指针
	 * @param octreeLoc 当前定位至Octree类的八叉树结构体指针
	 * @param point 给定三维点PointCloud::Point，搜寻基点
	 * @param radius 半径搜索半径
	 * @param &pointCloud 模板为PointCloud::Point的点云向量，储存搜寻点云结果
	 * @param &searchOctree 模板为Octree类的八叉树结构体指针的向量，储存搜寻八叉树结果最低一级的叶子节点
	 * @return void
	 */
	void radiusSearch(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	/**
	 * @brief radius search智能模式，或者块体模式
	 * @param octreeOG Octree类的根节点八叉树结构体指针
	 * @param octreeLoc 当前定位至Octree类的八叉树结构体指针
	 * @param point 给定三维点PointCloud::Point，搜寻基点
	 * @param radius 半径搜索半径
	 * @param &pointCloud 模板为PointCloud::Point的点云向量，储存搜寻八叉树叶子节点中心点坐标，层级为getLevel - 5或者1
	 * @return void
	 */
	void radiusSearchCube(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud);

	/**
	 * @brief radius search自由模式，或者块体模式
	 * @param octreeOG Octree类的根节点八叉树结构体指针
	 * @param octreeLoc 当前定位至Octree类的八叉树结构体指针
	 * @param point 给定三维点PointCloud::Point，搜寻基点
	 * @param radius 半径搜索半径
	 * @param &pointCloud 模板为PointCloud::Point的点云向量，储存搜寻八叉树叶子节点中心点坐标，层级为inputLevel（radiusSearchCubeModeByLevel）若inputLevel > locLevel则输出为空
	 * @return void
	 */
	void radiusSearchCubeByLevel(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, int inputLevel);

	/**
	 * @brief quadtree max搜寻方格z极大值，以及体素均值
	 * @param octreeOG Octree类的根节点八叉树结构体指针
	 * @param &qMaxPointCloud 模板为PointCloud::Point的点云向量，储存搜寻四叉树结果最低一级的叶子节点zMax时的坐标
	 * @param &minx 八叉树范围最小x
	 * @param &minx 八叉树范围最小y
	 * @return void
	 */
	void gpuQuadtreeMax(Octree::Octree_Struct octreeOG, std::vector<PointCloud::Point>& qMaxPointCloud, double minx, double miny);

	/**
	 * @brief octree 搜寻每个规定大小体素xyz均值
	 * @param octreeOG Octree类的根节点八叉树结构体指针
	 * @param &vMeanPointCloud 模板为PointCloud::Point的点云向量，储存搜寻八叉树结果最低一级的叶子节点所有点均值坐标
	 * @return void
	 */
	void gpuVoxelMean(Octree::Octree_Struct octreeWatershed, std::vector<PointCloud::Point>& vMeanPointCloud);
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

	//Brief：gpu计算每个体素的最大z值对应坐标
	void gpuMaxZvector(std::vector<PointCloud::Point> voxelPoint, PointCloud::Point& point);
};

