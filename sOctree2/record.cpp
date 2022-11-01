#include "Octree.cuh"
#include "Utility.cuh"
#include <math.h>
#include "Visualization.cuh"

/*
��־��
�汾��v004m
Brief������ĿΪ�˲���All in one prj��
�˲����ڼ����ƴ���30-60s
����ʱ�䣺2022/4/27
2022/4/27�������˰˲�������
2022/4/28�������radiusSearch���ܣ��޸�center��ʼ������ֵʱ����
�����޸���Octree�ഴ��+12s������ʱ��50s����
���޸�bug����radius>���������lengthʱ�����ֿ�ָ�뱨��
2022/4/29������radiusSearch����
�޸�bug��radiusSearch�����ظ����⣬δ����ԭʼoctree
�����˸���radiusѡ��level����
�����޸���ָ��bug
�������������ɾ���ս�㺯�����������cube�޷��պð�������ʱ������ֿսڵ㣬��ʱ�ýڵ�ָ��octreeΪ
Nullptr�����Ѽ�ʱ�ᱨ��
״̬���ѽ��
�ƻ���double����Ϊfloat����С�ڴ�ռ�ã����������ٶȣ�����������PCL
2022/4/30������OpenGL��Visualization��
�����vector<PointCloud::Point>��Matrix��ת��
��PointCloud::Point�ṹ��������������ݣ�a
����ĵã��ڹ������ļ��еĽṹ������#pragma once�ں��������л�ֱ�ӵ���cache�����޸�ʧЧ��
��Ҫɾ��projName->x64->Debug�ļ����е������ļ�
2022/5/1��ת��Utility��ΪCUDA��
2022/5/2������radiusSearch�߼������ټ��������ӿ��˼���Ч�ʣ�
�ȶԼ���Ч�ʣ�
���ȵ���size200��
v003a��27.043000 s 
v004a��23.900000 s
���ȳ������size500 with radius = 233��
v003a��35.469000 s	Level = 7 radiusSearchPoints: 15013710 radiusSearchOctree: 15013710 sPtcMat: 15013710 ��������ʱ�䣺4.826000 s
v004a��116.417000 s Level = 7 radiusSearchPoints: 15013710 radiusSearchOctree: 15013710 sPtcMat: 15013710 ��������ʱ�䣺14.737000 s
���߼�������ͬ����Ӧ��ͬ�����ܶ�����£�levelԽ��Ч�����������ԡ�����������ܶ�Խ��Ч��Խ���ԣ�
PointCloudLibrary��Neighbors within radius search at (50 50 50) with radius = 233
Points: 139864453 
��������ʱ�䣺250.311000 s
2022/5/4���Ż�radiusSearch����ģʽ����������Ч�ʼ���
���ӵ�����ת
2022/5/6��ʵ�ֿ���ģʽ���ĵ���8��������תMatrix��ʵ�ֵ�����תƽ��
2022/5/7��ʵ�ֵ���Ͷ���ͼ
����ͶӰͼ�ȸ��ã�https://zhuanlan.zhihu.com/p/73034007
����CUDA VBO��ʾ
2022/5/9�����ɾ�̬�⣬׼��Qt������OpenGL
2022/5/13������Octree�⹦��radiusSearchByLevel
2022/5/14���޸��˿տ���Ȼ��ʾbug
2022/5/17���ش����
����genMutiOctree�������ߴ����
���4400�ֱ���Ϊ10�Ŀ��壬��������ʱ�䣺58.697000 s
�޸�genMutiOctree��ʾС�͵���bug
2022/5/18���Է������������ı䣬single��mix����
�������ԣ����޸�bug
2022/5/20���ⶨ�޸�genMutiOctree�����߼���������һ������
����ʱ����ɷ��ಢװ������
2022/5/21����PointCloud����Ϊ����aֵĬ��ΪNO_ATTRIBUTE
ͷ�ļ����ɻ������ã���PointCloudͷ�ļ�������Utility.cuh�ᱨ��
�����������PointCloud.cpp�ļ�������
�Ƴ�OpenGL
msvcrt.lib C�������п���cuda���п��ͻ������Debugģʽ����ʧ��
2022/5/24���޸�genMutiOctree�߼�Bug
2022/9/30������sOctree2/Utility�����CHM��Ԫ��ֵȷ����Ncut���ص�Ԫ��ֵȷ����׼����ֵ������ʹ��˫���������Σ������ֵ
2022/10/5��Ϊ�����д󲿷ֺ������ע�ͣ����¼����߼���׼������ioStyle��
2022/10/6�����sOctree2/Visualization������Ӽ���CUDA���롣����sOctree2/PointCloudÿ������ȫ��double
2022/10/11������sOctree2/Visualization����Matrix��ʼ������
2022/10/14��sOctree2/Utility�����¶�˲����㷨�߼�
2022/10/30���޸�һ��sOctree2/Visualization��matrixMul�������
*/
