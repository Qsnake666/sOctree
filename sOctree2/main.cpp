#include "Octree.cuh"
#include "Utility.cuh"
#include <math.h>
#include "Visualization.cuh"

/*
日志：
版本：v004i
Brief：该项目为八叉树All in one prj。
八叉树亿级点云创建30-60s
创建时间：2022/4/27
2022/4/27：完善了八叉树功能
2022/4/28：完成了radiusSearch功能，修复center初始化及赋值时错误
问题修复后Octree类创建+12s，创建时间50s左右
待修复bug：当radius>最大立方体length时，出现空指针报错
2022/4/29：完善radiusSearch功能
修复bug：radiusSearch搜索重复问题，未引入原始octree
完善了根据radius选择level功能
正在修复空指针bug
解决方案：禁用删除空结点函数。当最外层cube无法刚好包裹点云时，会出现空节点，此时该节点指针octree为
Nullptr，在搜集时会报错
状态：已解决
计划将double更换为float，减小内存占用，增加运算速度，现在稍慢于PCL
2022/4/30：引入OpenGL和Visualization类
完成了vector<PointCloud::Point>到Matrix的转换
在PointCloud::Point结构中添加了属性数据：a
解决心得：在关联类文件中的结构体由于#pragma once在后续编译中会直接调用cache导致修改失效，
需要删除projName->x64->Debug文件夹中的所有文件
2022/5/1：转换Utility库为CUDA库
2022/5/2：更新radiusSearch逻辑，减少计算量，加快了计算效率！
比对计算效率：
均匀点云size200：
v003a：27.043000 s 
v004a：23.900000 s
均匀超大点云size500 with radius = 233：
v003a：35.469000 s	Level = 7 radiusSearchPoints: 15013710 radiusSearchOctree: 15013710 sPtcMat: 15013710 程序运行时间：4.826000 s
v004a：116.417000 s Level = 7 radiusSearchPoints: 15013710 radiusSearchOctree: 15013710 sPtcMat: 15013710 程序运行时间：14.737000 s
两者计算结果相同，反应相同点云密度情况下，level越高效率提升更明显。可推理点云密度越高效率越明显！
PointCloudLibrary：Neighbors within radius search at (50 50 50) with radius = 233
Points: 139864453 
程序运行时间：250.311000 s
2022/5/3：谁不知道并行计算呢？我真是搞笑，熟不知竟然CPU也能并发？
2022/5/4：优化radiusSearch块体模式，块体搜索效率极高
增加点云旋转
2022/5/6：实现块体模式中心点云8方向延申转Matrix，实现点云旋转平移
2022/5/7：实现点云投射绘图
总结：
开始没头绪，干着急；中期有头绪，进度快；快做完了就摆烂
点云投影图先搁置：https://zhuanlan.zhihu.com/p/73034007
测试CUDA VBO显示
2022/5/9：生成静态库，准备Qt中配置OpenGL

2022/5/13：更新Octree库功能radiusSearchByLevel
2022/5/14：修复了空块仍然显示bug
2022/5/17：重大更新
更新genMutiOctree适配更大尺寸点云
跨度4400分辨率为10的块体，程序运行时间：58.697000 s
修复genMutiOctree显示小型点云bug
2022/5/18：对方块属性做出改变，single和mix两类
更改属性，并修复bug
2022/5/20：拟定修改genMutiOctree部分逻辑，仅遍历一次数组
遍历时候完成分类并装入容器
2022/5/21：将PointCloud类中为定义a值默认为NO_ATTRIBUTE
头文件不可互相引用，在PointCloud头文件中引用Utility.cuh会报错
解决方案，在PointCloud.cpp文件中引入
移除OpenGL
msvcrt.lib C语言运行库与cuda运行库冲突，导致Debug模式运行失败
2022/5/24：修复genMutiOctree逻辑Bug
*/

//调用实验
int main(int argc, char* argv[])
{
	double duration;
	int size = 200;
	std::vector<PointCloud::Point> pointCloud;
	
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				pointCloud.push_back(PointCloud::Point((long long)i * 10, (long long)j * 40, k, i));
			}
		}
	}
	
	//建立八叉树模型模块
	//time_t startAll = clock();
	/*
	2400 2400 304.5
2400 2400 336
2000 1200 8
2000 1200 24
2000 1200 40
2000 1200 152
2000 1200 248
2000 1200 264
2000 1200 280
	
	pointCloud.push_back(PointCloud::Point(2400, 2400, 336, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 8, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 24, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 40, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 152, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 248, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 264, 1));
	pointCloud.push_back(PointCloud::Point(2000, 1200, 280, 1));
	*/
	int resolution = 10;
	Octree* oct = new Octree;
	Utility* uty = new Utility;

	std::vector < Octree*> octMul;
	time_t starta = clock();
	uty->genMultiOctree(octMul, pointCloud, resolution);
	time_t enda = clock();
	std::cout << "多Octree数量" << octMul.size() << std::endl;
	duration = enda - starta;
	printf("程序运行时间：%f s \n", duration / 1000);
	/*
	time_t startb = clock();
	uty->genOctree(oct, pointCloud, resolution);//只要随机数种子不变，其生成的随机数序列就不会改变。
	time_t endb = clock();
	duration = endb - startb;
	printf("程序运行时间：%f s \n", duration / 1000);
	*/

	//查找点位置 
	int level, level2;
	time_t startAll = clock();
	Octree::Octree_Struct ost; //实体
	Octree::Octree_Struct ost2; //实体
	PointCloud::Point point = PointCloud::Point(0,0,0);
	for (int i = 1; i < 240; i+=10) {//压力测试
		time_t start = clock();
		double radius = (long long)3 + i;
		//level = uty->getLevel(oct->octree, radius, resolution);
		level2 = uty->getLevel(octMul[0]->octree, radius, resolution);
		//std::cout << "Level = " << level << std::endl;
		std::cout << "Level2 = " << level2 << std::endl;

		//查找点在某一层octree意义：在radius搜寻过程中根据radius快速定位
		//这样cube外元素寻找仅需要先确定相邻center点，查询center点在该level的octree即可
		//ost = uty->locatePoints(oct->octree, point, level);
		ost2 = uty->locatePoints(octMul[0]->octree, point, level2);
		//查询时间 0s

		//查找点radius范围点和节点
		/*
		std::vector<PointCloud::Point> searchPointCloud;
		std::vector<Octree::Octree_Struct> searchOctree;
		uty->radiusSearch(oct->octree, ost, point, radius, searchPointCloud, searchOctree);
		std::cout << "radiusSearchPoints: " << searchPointCloud.size() << std::endl;
		std::cout << "radiusSearchOctree: " << searchOctree.size() << std::endl;
		*/
		
		//快速搜寻块体缓存 共计时间0.09s
		//std::vector<PointCloud::Point> searchCubeCenter;
		std::vector<PointCloud::Point> searchCubeCenter2;
		//uty->radiusSearchCube(oct->octree, ost, point, radius, searchCubeCenter);
		uty->radiusSearchCube(octMul[0]->octree, ost2, point, radius, searchCubeCenter2);
		//std::cout << "radiusSearchCube: " << searchCubeCenter.size() << std::endl;
		std::cout << "radiusSearchCube: " << searchCubeCenter2.size() << std::endl;

		//得到块体模式Cube长度 创建函数getLevelCube
		int levelCube = level2 - 4;
		if (levelCube < 1)
			levelCube = 1;
		double lengthCube = pow(2, levelCube - 1) * resolution;

		//可视化模块
		Visualization* vis = new Visualization;
		Visualization::Matrix sPtcMat;
		Visualization::Matrix sPtcMat2;
		Visualization::Matrix sPtcMatT; //转置
		Visualization::Matrix outMat; //输出
		//点云模式转Matrix
		//vis->vectorToMatrix(searchCubeCenter, &sPtcMat);
		//块体模式转Matrix补齐8方向
		/*
		vis->vectorToMatrixCube(searchCubeCenter, &sPtcMat, lengthCube);
		vis->vectorToMatrixCube(searchCubeCenter2, &sPtcMat2, lengthCube);
		std::cout << "sPtcMat: " << sPtcMat.height << std::endl;
		std::cout << "sPtcMat2: " << sPtcMat2.height << std::endl;
		//for (int i = 0; i < sPtcMat2.height; i += 9) {
			//std::cout << sPtcMat2.elements[i * sPtcMat2.stride + 3] << std::endl;
		//}
		//点云旋转平移模块
		Visualization::Matrix R;
		vis->initMatrix(&R, 4, 4);
		double rad = 3.1415926 / 2; //90°
		vis->genRotateMatrix(R, 'x', rad);
		double tx, ty, tz;
		tx = 10; ty = 0; tz = 0;
		vis->genTranslationMatrix(R, tx, ty, tz);
		//原Matrix转置 R*A'
		vis->initMatrix(&sPtcMatT, sPtcMat.width, sPtcMat.height);
		vis->matrixTranspose(sPtcMat, sPtcMatT);
		//旋转平移
		outMat = vis->ptcRotate(R, sPtcMatT);
		*/
		time_t end = clock();
		duration = end - start;
		printf("程序运行时间：%f s \n", duration / 1000);
	}
	time_t endAll = clock();
	duration = endAll - startAll;
	printf("程序运行时间：%f s \n", duration / 1000);
}