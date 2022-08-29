#include "Octree.cuh"
#include "Utility.cuh"
#include <math.h>
#include "Visualization.cuh"

/*
��־��
�汾��v004i
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
2022/5/3��˭��֪�����м����أ������Ǹ�Ц���첻֪��ȻCPUҲ�ܲ�����
2022/5/4���Ż�radiusSearch����ģʽ����������Ч�ʼ���
���ӵ�����ת
2022/5/6��ʵ�ֿ���ģʽ���ĵ���8��������תMatrix��ʵ�ֵ�����תƽ��
2022/5/7��ʵ�ֵ���Ͷ���ͼ
�ܽ᣺
��ʼûͷ�������ż���������ͷ�������ȿ죻�������˾Ͱ���
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
*/

//����ʵ��
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
	
	//�����˲���ģ��ģ��
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
	std::cout << "��Octree����" << octMul.size() << std::endl;
	duration = enda - starta;
	printf("��������ʱ�䣺%f s \n", duration / 1000);
	/*
	time_t startb = clock();
	uty->genOctree(oct, pointCloud, resolution);//ֻҪ��������Ӳ��䣬�����ɵ���������оͲ���ı䡣
	time_t endb = clock();
	duration = endb - startb;
	printf("��������ʱ�䣺%f s \n", duration / 1000);
	*/

	//���ҵ�λ�� 
	int level, level2;
	time_t startAll = clock();
	Octree::Octree_Struct ost; //ʵ��
	Octree::Octree_Struct ost2; //ʵ��
	PointCloud::Point point = PointCloud::Point(0,0,0);
	for (int i = 1; i < 240; i+=10) {//ѹ������
		time_t start = clock();
		double radius = (long long)3 + i;
		//level = uty->getLevel(oct->octree, radius, resolution);
		level2 = uty->getLevel(octMul[0]->octree, radius, resolution);
		//std::cout << "Level = " << level << std::endl;
		std::cout << "Level2 = " << level2 << std::endl;

		//���ҵ���ĳһ��octree���壺��radius��Ѱ�����и���radius���ٶ�λ
		//����cube��Ԫ��Ѱ�ҽ���Ҫ��ȷ������center�㣬��ѯcenter���ڸ�level��octree����
		//ost = uty->locatePoints(oct->octree, point, level);
		ost2 = uty->locatePoints(octMul[0]->octree, point, level2);
		//��ѯʱ�� 0s

		//���ҵ�radius��Χ��ͽڵ�
		/*
		std::vector<PointCloud::Point> searchPointCloud;
		std::vector<Octree::Octree_Struct> searchOctree;
		uty->radiusSearch(oct->octree, ost, point, radius, searchPointCloud, searchOctree);
		std::cout << "radiusSearchPoints: " << searchPointCloud.size() << std::endl;
		std::cout << "radiusSearchOctree: " << searchOctree.size() << std::endl;
		*/
		
		//������Ѱ���建�� ����ʱ��0.09s
		//std::vector<PointCloud::Point> searchCubeCenter;
		std::vector<PointCloud::Point> searchCubeCenter2;
		//uty->radiusSearchCube(oct->octree, ost, point, radius, searchCubeCenter);
		uty->radiusSearchCube(octMul[0]->octree, ost2, point, radius, searchCubeCenter2);
		//std::cout << "radiusSearchCube: " << searchCubeCenter.size() << std::endl;
		std::cout << "radiusSearchCube: " << searchCubeCenter2.size() << std::endl;

		//�õ�����ģʽCube���� ��������getLevelCube
		int levelCube = level2 - 4;
		if (levelCube < 1)
			levelCube = 1;
		double lengthCube = pow(2, levelCube - 1) * resolution;

		//���ӻ�ģ��
		Visualization* vis = new Visualization;
		Visualization::Matrix sPtcMat;
		Visualization::Matrix sPtcMat2;
		Visualization::Matrix sPtcMatT; //ת��
		Visualization::Matrix outMat; //���
		//����ģʽתMatrix
		//vis->vectorToMatrix(searchCubeCenter, &sPtcMat);
		//����ģʽתMatrix����8����
		/*
		vis->vectorToMatrixCube(searchCubeCenter, &sPtcMat, lengthCube);
		vis->vectorToMatrixCube(searchCubeCenter2, &sPtcMat2, lengthCube);
		std::cout << "sPtcMat: " << sPtcMat.height << std::endl;
		std::cout << "sPtcMat2: " << sPtcMat2.height << std::endl;
		//for (int i = 0; i < sPtcMat2.height; i += 9) {
			//std::cout << sPtcMat2.elements[i * sPtcMat2.stride + 3] << std::endl;
		//}
		//������תƽ��ģ��
		Visualization::Matrix R;
		vis->initMatrix(&R, 4, 4);
		double rad = 3.1415926 / 2; //90��
		vis->genRotateMatrix(R, 'x', rad);
		double tx, ty, tz;
		tx = 10; ty = 0; tz = 0;
		vis->genTranslationMatrix(R, tx, ty, tz);
		//ԭMatrixת�� R*A'
		vis->initMatrix(&sPtcMatT, sPtcMat.width, sPtcMat.height);
		vis->matrixTranspose(sPtcMat, sPtcMatT);
		//��תƽ��
		outMat = vis->ptcRotate(R, sPtcMatT);
		*/
		time_t end = clock();
		duration = end - start;
		printf("��������ʱ�䣺%f s \n", duration / 1000);
	}
	time_t endAll = clock();
	duration = endAll - startAll;
	printf("��������ʱ�䣺%f s \n", duration / 1000);
}