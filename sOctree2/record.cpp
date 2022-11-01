#include "Octree.cuh"
#include "Utility.cuh"
#include <math.h>
#include "Visualization.cuh"

/*
日志：
版本：v004m
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
2022/5/4：优化radiusSearch块体模式，块体搜索效率极高
增加点云旋转
2022/5/6：实现块体模式中心点云8方向延申转Matrix，实现点云旋转平移
2022/5/7：实现点云投射绘图
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
2022/9/30：更新sOctree2/Utility，完成CHM单元极值确定，Ncut体素单元均值确定，准备插值工作，使用双立方（三次）卷积插值
2022/10/5：为工程中大部分函数添加注释，更新计算逻辑，准备引入ioStyle类
2022/10/6：添加sOctree2/Visualization，矩阵加减法CUDA代码。更改sOctree2/PointCloud每个属性全是double
2022/10/11：更新sOctree2/Visualization关于Matrix初始化函数
2022/10/14：sOctree2/Utility，更新多八叉树算法逻辑
2022/10/30：修复一个sOctree2/Visualization种matrixMul代码错误
*/
