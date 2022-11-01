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
//ʵ��Octree�����࣬OctreeΪ�丸��

#define NO_ATTRIBUTE -1
#define MIX_ATTRIBUTE 0

class Utility :public Octree
{
public:
	/**
	 * @brief ���ɵ��ư˲�������
	 * @param *oct Octree���ʵ��
	 * @param pointCloud ģ��ΪPointCloud::Point�ĵ�������
	 * @param resolution ���ɰ˲����������һ����Ҷ�ӽڵ�ĳߴ�
	 * @return void
	 */
	void genOctree(Octree* oct, std::vector<PointCloud::Point> pointCloud, double resolution);

	/**
	 * @brief ���ɶ�����ư˲�������
	 * @param &octMul ģ��ΪOctree���ʵ��������
	 * @param pointCloud ģ��ΪPointCloud::Point�ĵ�������
	 * @param resolution ���ɰ˲����������һ����Ҷ�ӽڵ�ĳߴ�
	 * @return void
	 */
	void genMultiOctree(std::vector < Octree*>& octMul, std::vector<PointCloud::Point> pointCloud, double resolution);

	/**
	 * @brief �뾶����ʱ���ݰ뾶�Զ������Ӧ�˲�����Ѱlevel
	 * @param octreeOG Octree��ĸ��ڵ�˲����ṹ��ָ��
	 * @param radius �뾶�����뾶
	 * @param resolution ���ɰ˲����������һ����Ҷ�ӽڵ�ĳߴ�
	 * @return int �˲����Ĳ㼶
	 */
	int getLevel(Octree::Octree_Struct octreeOG, double radius, double resolution);

	/**
	 * @brief �����㶨λ���涨��Ȱ˲����ṹ
	 * @param octree Octree��İ˲����ṹ��ָ��
	 * @param point ������ά��PointCloud::Point
	 * @param level �涨�˲����㼶
	 * @return Octree::Octree_Struct Octree��İ˲����ṹ��ָ��
	 */
	Octree::Octree_Struct locatePoints(Octree::Octree_Struct octree, PointCloud::Point point, int level);

	/**
	 * @brief radius search����ģʽ�����߾��ܿ���ģʽ
	 * @param octreeOG Octree��ĸ��ڵ�˲����ṹ��ָ��
	 * @param octreeLoc ��ǰ��λ��Octree��İ˲����ṹ��ָ��
	 * @param point ������ά��PointCloud::Point����Ѱ����
	 * @param radius �뾶�����뾶
	 * @param &pointCloud ģ��ΪPointCloud::Point�ĵ���������������Ѱ���ƽ��
	 * @param &searchOctree ģ��ΪOctree��İ˲����ṹ��ָ���������������Ѱ�˲���������һ����Ҷ�ӽڵ�
	 * @return void
	 */
	void radiusSearch(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree);

	/**
	 * @brief radius search����ģʽ�����߿���ģʽ
	 * @param octreeOG Octree��ĸ��ڵ�˲����ṹ��ָ��
	 * @param octreeLoc ��ǰ��λ��Octree��İ˲����ṹ��ָ��
	 * @param point ������ά��PointCloud::Point����Ѱ����
	 * @param radius �뾶�����뾶
	 * @param &pointCloud ģ��ΪPointCloud::Point�ĵ���������������Ѱ�˲���Ҷ�ӽڵ����ĵ����꣬�㼶ΪgetLevel - 5����1
	 * @return void
	 */
	void radiusSearchCube(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud);

	/**
	 * @brief radius search����ģʽ�����߿���ģʽ
	 * @param octreeOG Octree��ĸ��ڵ�˲����ṹ��ָ��
	 * @param octreeLoc ��ǰ��λ��Octree��İ˲����ṹ��ָ��
	 * @param point ������ά��PointCloud::Point����Ѱ����
	 * @param radius �뾶�����뾶
	 * @param &pointCloud ģ��ΪPointCloud::Point�ĵ���������������Ѱ�˲���Ҷ�ӽڵ����ĵ����꣬�㼶ΪinputLevel��radiusSearchCubeModeByLevel����inputLevel > locLevel�����Ϊ��
	 * @return void
	 */
	void radiusSearchCubeByLevel(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, int inputLevel);

	/**
	 * @brief quadtree max��Ѱ����z����ֵ���Լ����ؾ�ֵ
	 * @param octreeOG Octree��ĸ��ڵ�˲����ṹ��ָ��
	 * @param &qMaxPointCloud ģ��ΪPointCloud::Point�ĵ���������������Ѱ�Ĳ���������һ����Ҷ�ӽڵ�zMaxʱ������
	 * @param &minx �˲�����Χ��Сx
	 * @param &minx �˲�����Χ��Сy
	 * @return void
	 */
	void gpuQuadtreeMax(Octree::Octree_Struct octreeOG, std::vector<PointCloud::Point>& qMaxPointCloud, double minx, double miny);

	/**
	 * @brief octree ��Ѱÿ���涨��С����xyz��ֵ
	 * @param octreeOG Octree��ĸ��ڵ�˲����ṹ��ָ��
	 * @param &vMeanPointCloud ģ��ΪPointCloud::Point�ĵ���������������Ѱ�˲���������һ����Ҷ�ӽڵ����е��ֵ����
	 * @return void
	 */
	void gpuVoxelMean(Octree::Octree_Struct octreeWatershed, std::vector<PointCloud::Point>& vMeanPointCloud);
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

	//Brief��gpu����ÿ�����ص����zֵ��Ӧ����
	void gpuMaxZvector(std::vector<PointCloud::Point> voxelPoint, PointCloud::Point& point);
};

