#include "Utility.cuh"

void Utility::genOctree(Octree* oct, std::vector<PointCloud::Point> pointCloud, double resolution) {
	if (pointCloud.size() > 0) {
		oct->octLength = resolution;
		oct->setPoint(pointCloud);
		oct->CreatOctreeByPointCloud();
	}
}

void Utility::genMultiOctree(std::vector < Octree*>& octMul, std::vector<PointCloud::Point> pointCloud, double resolution) {
	maxValue(pointCloud, max_x, max_y, max_z);
	minValue(pointCloud, min_x, min_y, min_z);

	double length = octLength;
	double maxLength;
	double lengthX = max_x - min_x;
	double lengthY = max_y - min_y;
	double lengthZ = max_z - min_z;
	int level = 1;
	//maxLength为最小能包含所有点的边长
	maxLength = lengthX > lengthY ? lengthX : lengthY;
	maxLength = maxLength > lengthZ ? maxLength : lengthZ;

	double threshold = 30;

	if (maxLength / resolution > threshold) {
		//开始判断多八叉树延伸方向
		int idxA, idxB, idxC;
		double lftx, lfty, lftz;
		double lengthTmp;
		//X方向延伸
		if (lengthX > threshold * resolution) {
			idxA = lengthX / resolution / threshold;
			lftx = (long long)(lengthX / resolution) % (long long)threshold;
			lengthTmp = resolution * threshold;
			//Y方向延伸 Z方向不延伸
			if (lengthY > threshold * resolution && lengthZ <= threshold * resolution); {
				idxB = lengthY / resolution / threshold;
				lfty = (long long)(lengthY / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						std::vector<PointCloud::Point> pointCloudTmp;
						if (i != idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i != idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
									pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
									pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						if (pointCloudTmp.size() > 0) {
							Octree* oct = new Octree;
							oct->octLength = resolution;
							oct->setPoint(pointCloudTmp);
							oct->CreatOctreeByPointCloud();
							octMul.push_back(oct);
						}
					}
				}
			}
			//Y方向不延伸 Z方向延伸 lengthTmp没有修改
			if (lengthY <= threshold * resolution && lengthZ > threshold * resolution) {
				std::cout << threshold * resolution << std::endl;
				idxB = lengthZ / resolution / threshold;
				lftz = (long long)(lengthZ / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						std::vector<PointCloud::Point> pointCloudTmp;
						if (i != idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].z > min_z + j * lengthTmp && pointCloud[m].z <= min_z + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i != idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].z > min_z + ((long long)j - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + j * lengthTmp + lftz * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
									pointCloud[m].z > min_z + j * lengthTmp && pointCloud[m].z <= min_z + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
									pointCloud[m].z > min_z + ((long long)j - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + j * lengthTmp + lftz * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						if (pointCloudTmp.size() > 0) {
							Octree* oct = new Octree;
							oct->octLength = resolution;
							oct->setPoint(pointCloudTmp);
							oct->CreatOctreeByPointCloud();
							octMul.push_back(oct);
						}
					}
				}
			}
			//Y方向延伸 Z方向延伸 lengthTmp没有修改
			if (lengthY > threshold * resolution && lengthZ > threshold * resolution) {
				idxB = lengthY / resolution / threshold;
				lfty = (long long)(lengthY / resolution) % (long long)threshold;
				idxC = lengthZ / resolution / threshold;
				lftz = (long long)(lengthZ / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						for (int k = 0; k < idxC + 1; k++) {
							std::vector<PointCloud::Point> pointCloudTmp;
							if (i != idxA && j != idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j == idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j != idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j == idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j != idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution){
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j == idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j != idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j == idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							if (pointCloudTmp.size() > 0) {
								Octree* oct = new Octree;
								oct->octLength = resolution;
								oct->setPoint(pointCloudTmp);
								oct->CreatOctreeByPointCloud();
								octMul.push_back(oct);
							}
						}
					}
				}
			}
			//Y方向不延伸 Z方向不延伸
			if (lengthY <= threshold * resolution && lengthZ <= threshold * resolution) {
				for (int i = 0; i < idxA + 1; i++) {
					std::vector<PointCloud::Point> pointCloudTmp;
					if (i != idxA) {
						for (int m = pointCloud.size() - 1; m >= 0; m--) {
							if (pointCloud[m].x > min_x + i * lengthTmp && pointCloud[m].x <= min_x + ((long long)i + 1) * lengthTmp) {
								pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
							}
						}
					}
					if (i == idxA) {
						for (int m = pointCloud.size() - 1; m >= 0; m--) {
							if (pointCloud[m].x > min_x + ((long long)i - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + i * lengthTmp + lftx * resolution) {
								pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
							}
						}
					}
					if (pointCloudTmp.size() > 0) {
						Octree* oct = new Octree;
						oct->octLength = resolution;
						oct->setPoint(pointCloudTmp);
						oct->CreatOctreeByPointCloud();
						octMul.push_back(oct);
					}
				}
			}
		}
		//Y方向延伸
		else if (lengthY > threshold * resolution) {
			idxA = lengthY / resolution / threshold;
			lfty = (long long)(lengthY / resolution) % (long long)threshold;
			lengthTmp = resolution * threshold;
			//X方向延伸 Z方向不延伸
			if (lengthX > threshold * resolution && lengthZ <= threshold * resolution); {
				idxB = lengthX / resolution / threshold;
				lftx = (long long)(lengthX / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						std::vector<PointCloud::Point> pointCloudTmp;
						if (i != idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i != idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
									pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
									pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						if (pointCloudTmp.size() > 0) {
							Octree* oct = new Octree;
							oct->octLength = resolution;
							oct->setPoint(pointCloudTmp);
							oct->CreatOctreeByPointCloud();
							octMul.push_back(oct);
						}
					}
				}
			}
			//X方向不延伸 Z方向延伸 lengthTmp没有修改
			if (lengthX <= threshold * resolution && lengthZ > threshold * resolution) {
				idxB = lengthZ / resolution / threshold;
				lftz = (long long)(lengthZ / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						std::vector<PointCloud::Point> pointCloudTmp;
						if (i != idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].z > min_z + j * lengthTmp && pointCloud[m].z <= min_z + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i != idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].z > min_z + ((long long)j - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + j * lengthTmp + lftz * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
									pointCloud[m].z > min_z + j * lengthTmp && pointCloud[m].z <= min_z + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
									pointCloud[m].z > min_z + ((long long)j - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + j * lengthTmp + lftz * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						if (pointCloudTmp.size() > 0) {
							Octree* oct = new Octree;
							oct->octLength = resolution;
							oct->setPoint(pointCloudTmp);
							oct->CreatOctreeByPointCloud();
							octMul.push_back(oct);
						}
					}
				}
			}
			//X方向延伸 Z方向延伸 lengthTmp没有修改
			if (lengthX > threshold * resolution && lengthZ > threshold * resolution) {
				idxB = lengthX / resolution / threshold;
				lftx = (long long)(lengthX / resolution) % (long long)threshold;
				idxC = lengthZ / resolution / threshold;
				lftz = (long long)(lengthZ / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						for (int k = 0; k < idxC + 1; k++) {
							std::vector<PointCloud::Point> pointCloudTmp;
							if (i != idxA && j != idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j == idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j != idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j == idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution &&
										pointCloud[m].z > min_z + k * lengthTmp && pointCloud[m].z <= min_z + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j != idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j == idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j != idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j == idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution &&
										pointCloud[m].z > min_z + ((long long)k - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + k * lengthTmp + lftz * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							if (pointCloudTmp.size() > 0) {
								Octree* oct = new Octree;
								oct->octLength = resolution;
								oct->setPoint(pointCloudTmp);
								oct->CreatOctreeByPointCloud();
								octMul.push_back(oct);
							}
						}
					}
				}
			}
			//X方向不延伸 Z方向不延伸
			if (lengthX <= threshold * resolution && lengthZ <= threshold * resolution) {
				for (int i = 0; i < idxA + 1; i++) {
					std::vector<PointCloud::Point> pointCloudTmp;
					if (i != idxA) {
						for (int m = pointCloud.size() - 1; m >= 0; m--) {
							if (pointCloud[m].y > min_y + i * lengthTmp && pointCloud[m].y <= min_y + ((long long)i + 1) * lengthTmp) {
								pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
							}
						}
					}
					if (i == idxA) {
						for (int m = pointCloud.size() - 1; m >= 0; m--) {
							if (pointCloud[m].y > min_y + ((long long)i - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + i * lengthTmp + lfty * resolution) {
								pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
							}
						}
					}
					if (pointCloudTmp.size() > 0) {
						Octree* oct = new Octree;
						oct->octLength = resolution;
						oct->setPoint(pointCloudTmp);
						oct->CreatOctreeByPointCloud();
						octMul.push_back(oct);
					}
				}
			}
		}
		//Z方向延伸
		else if (lengthZ > threshold * resolution) {
			idxA = lengthZ / resolution / threshold;
			lftz = (long long)(lengthZ / resolution) % (long long)threshold;
			lengthTmp = resolution * threshold;
			//Y方向延伸 X方向不延伸
			if (lengthY > threshold * resolution && lengthX <= threshold * resolution); {
				idxB = lengthY / resolution / threshold;
				lfty = (long long)(lengthY / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						std::vector<PointCloud::Point> pointCloudTmp;
						if (i != idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i != idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
									pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
									pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						if (pointCloudTmp.size() > 0) {
							Octree* oct = new Octree;
							oct->octLength = resolution;
							oct->setPoint(pointCloudTmp);
							oct->CreatOctreeByPointCloud();
							octMul.push_back(oct);
						}
					}
				}
			}
			//Y方向不延伸 X方向延伸 lengthTmp没有修改
			if (lengthY <= threshold * resolution && lengthX > threshold * resolution) {
				idxB = lengthX / resolution / threshold;
				lftx = (long long)(lengthX / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						std::vector<PointCloud::Point> pointCloudTmp;
						if (i != idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp) {
								}
							}
						}
						else if (i != idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
									pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j != idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
									pointCloud[m].x > min_x + j * lengthTmp && pointCloud[m].x <= min_x + ((long long)j + 1) * lengthTmp) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						else if (i == idxA && j == idxB) {
							for (int m = pointCloud.size() - 1; m >= 0; m--) {
								if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
									pointCloud[m].x > min_x + ((long long)j - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + j * lengthTmp + lftx * resolution) {
									pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
								}
							}
						}
						if (pointCloudTmp.size() > 0) {
							Octree* oct = new Octree;
							oct->octLength = resolution;
							oct->setPoint(pointCloudTmp);
							oct->CreatOctreeByPointCloud();
							octMul.push_back(oct);
						}
					}
				}
			}
			//Y方向延伸 X方向延伸 lengthTmp没有修改
			if (lengthY > threshold * resolution && lengthX > threshold * resolution) {
				idxB = lengthY / resolution / threshold;
				lfty = (long long)(lengthY / resolution) % (long long)threshold;
				idxC = lengthX / resolution / threshold;
				lftz = (long long)(lengthX / resolution) % (long long)threshold;
				lengthTmp = resolution * threshold;
				for (int i = 0; i < idxA + 1; i++) {
					for (int j = 0; j < idxB + 1; j++) {
						for (int k = 0; k < idxC + 1; k++) {
							std::vector<PointCloud::Point> pointCloudTmp;
							if (i != idxA && j != idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].x > min_x + k * lengthTmp && pointCloud[m].x <= min_x + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j == idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + k * lengthTmp && pointCloud[m].x <= min_x + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j != idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].x > min_x + k * lengthTmp && pointCloud[m].x <= min_x + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j == idxB && k != idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + k * lengthTmp && pointCloud[m].x <= min_x + ((long long)k + 1) * lengthTmp) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j != idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].x > min_x + ((long long)k - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + k * lengthTmp + lftx * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i != idxA && j == idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + ((long long)k - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + k * lengthTmp + lftx * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j != idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
										pointCloud[m].y > min_y + j * lengthTmp && pointCloud[m].y <= min_y + ((long long)j + 1) * lengthTmp &&
										pointCloud[m].x > min_x + ((long long)k - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + k * lengthTmp + lftx * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							else if (i == idxA && j == idxB && k == idxC) {
								for (int m = pointCloud.size() - 1; m >= 0; m--) {
									if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution &&
										pointCloud[m].y > min_y + ((long long)j - 1) * lengthTmp + lfty * resolution && pointCloud[m].y <= min_y + j * lengthTmp + lfty * resolution &&
										pointCloud[m].x > min_x + ((long long)k - 1) * lengthTmp + lftx * resolution && pointCloud[m].x <= min_x + k * lengthTmp + lftx * resolution) {
										pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
									}
								}
							}
							if (pointCloudTmp.size() > 0) {
								Octree* oct = new Octree;
								oct->octLength = resolution;
								oct->setPoint(pointCloudTmp);
								oct->CreatOctreeByPointCloud();
								octMul.push_back(oct);
							}
						}
					}
				}
			}
			//Y方向不延伸 X方向不延伸
			if (lengthY <= threshold * resolution && lengthX <=threshold * resolution) {
				for (int i = 0; i < idxA + 1; i++) {
					std::vector<PointCloud::Point> pointCloudTmp;
					if (i != idxA) {
						for (int m = pointCloud.size() - 1; m >= 0; m--) {
							if (pointCloud[m].z > min_z + i * lengthTmp && pointCloud[m].z <= min_z + ((long long)i + 1) * lengthTmp) {
								pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
							}
						}
					}
					if (i == idxA) {
						for (int m = pointCloud.size() - 1; m >= 0; m--) {
							if (pointCloud[m].z > min_z + ((long long)i - 1) * lengthTmp + lftz * resolution && pointCloud[m].z <= min_z + i * lengthTmp + lftz * resolution) {
								pointCloudTmp.push_back(PointCloud::Point(pointCloud[m].x, pointCloud[m].y, pointCloud[m].z, pointCloud[m].a));
							}
						}
					}
					if (pointCloudTmp.size() > 0) {
						Octree* oct = new Octree;
						oct->octLength = resolution;
						oct->setPoint(pointCloudTmp);
						oct->CreatOctreeByPointCloud();
						octMul.push_back(oct);
					}
				}
			}
		}
		//不延申
		else if (lengthX <= threshold * resolution && lengthY <= threshold * resolution && lengthZ <= threshold * resolution) {
			if (pointCloud.size() > 0) {
				Octree* oct = new Octree;
				oct->octLength = resolution;
				oct->setPoint(pointCloud);
				oct->CreatOctreeByPointCloud();
				octMul.push_back(oct);
			}
		}
	}
	//不延申
	else if (lengthX <= threshold * resolution && lengthY <= threshold * resolution && lengthZ <= threshold * resolution) {
		if (pointCloud.size() > 0) {
			Octree* oct = new Octree;
			oct->octLength = resolution;
			oct->setPoint(pointCloud);
			oct->CreatOctreeByPointCloud();
			octMul.push_back(oct);
		}
	}
}

Octree::Octree_Struct Utility::locatePoints(Octree::Octree_Struct octree, PointCloud::Point point, int level) {
	/*
	//预处理level是否合格
	if (level > octree->level)
		level = octree->level;
	*/
	if (octree->level == level) {
		return octree;
	}
	if (point.x >= octree->center.x && point.y >= octree->center.y && point.z >= octree->center.z) {
		locatePoints(octree->nodes[0], point, level);
	}
	else if (point.x < octree->center.x && point.y >= octree->center.y && point.z >= octree->center.z) {
		locatePoints(octree->nodes[1], point, level);
	}
	else if (point.x >= octree->center.x && point.y < octree->center.y && point.z >= octree->center.z) {
		locatePoints(octree->nodes[2], point, level);
	}
	else if (point.x < octree->center.x && point.y < octree->center.y && point.z >= octree->center.z) {
		locatePoints(octree->nodes[3], point, level);
	}
	else if (point.x >= octree->center.x && point.y >= octree->center.y && point.z < octree->center.z) {
		locatePoints(octree->nodes[4], point, level);
	}
	else if (point.x < octree->center.x && point.y >= octree->center.y && point.z < octree->center.z) {
		locatePoints(octree->nodes[5], point, level);
	}
	else if (point.x >= octree->center.x && point.y < octree->center.y && point.z < octree->center.z) {
		locatePoints(octree->nodes[6], point, level);
	}
	else if (point.x < octree->center.x && point.y < octree->center.y && point.z < octree->center.z) {
		locatePoints(octree->nodes[7], point, level);
	}
}

int Utility::getLevel(Octree::Octree_Struct octreeOG, double radius, double resolution) {
	/*
	参考公式：pow(2,level)/2>radius/octLength
	level>log2(radius*2/octLength)
	*/
	int level;
	if ((log2(2 * radius / resolution)) - int(log2(2 * radius / resolution)) == 0.0)
		level = log2(2 * radius / resolution) + 1;
	else
		level = log2(2 * radius / resolution) + 2;
	if (level > octreeOG->level)
		level = octreeOG->level;
	return level;
}

void Utility::getPointOctree(Octree::Octree_Struct octree, std::vector<Octree::Octree_Struct>& vOctree) {
	if (octree->level == 1) {//防止到达边界外报错需要禁用清理空节点功能
		if (octree->points.size() > 0)
			vOctree.push_back(octree);
		return;
	}
	for (int i = 0; i < 8; i++) {
		getPointOctree(octree->nodes[i], vOctree);
	}
}

std::vector<Octree::Octree_Struct> Utility::getLowerOctree(Octree::Octree_Struct octree, PointCloud::Point point, double radius) {
	std::vector<Octree::Octree_Struct> vOctree;
	std::vector<Octree::Octree_Struct> vOctree2;
	if (octree->level >= 2) {
		for (int i = 0; i < 8; i++) {
			vOctree.push_back(octree->nodes[i]);
		}
		for (int k = 0; k < 8; k++) {
			double oDis = pow((point.x - vOctree[k]->center.x), 2) + pow((point.y - vOctree[k]->center.y), 2) + pow((point.z - vOctree[k]->center.z), 2);
			double cDis = 3.0 / 4.0 * pow(vOctree[k]->length, 2);
			if (pow(oDis,0.5) < radius + pow(cDis,0.5)) {
				vOctree2.push_back(vOctree[k]);
			}
		}
		return vOctree2;
	}
	else
		return vOctree2;
}

void Utility::radiusSearch(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree) {
	//SearchMode1：pointXYZ + radius < octree.CenterXYZ 即可以以octree为顶层向下寻找点，寻找octree这个立方体内点
	if ((point.x + radius) < (octreeLoc->center.x + octreeLoc->length / 2) &&
		(point.x - radius) > (octreeLoc->center.x - octreeLoc->length / 2) &&
		(point.y + radius) < (octreeLoc->center.y + octreeLoc->length / 2) &&
		(point.y - radius) > (octreeLoc->center.y - octreeLoc->length / 2) &&
		(point.z + radius) < (octreeLoc->center.z + octreeLoc->length / 2) &&
		(point.z - radius) > (octreeLoc->center.z - octreeLoc->length / 2)) {
		radiusSearchMode1(octreeLoc, point, radius, pointCloud, searchOctree);
	}
	//SearchMode2：pointXYZ + radius在octree立方体外
	else {
		if (octreeLoc->level != 1) {
			//Level > 1模式
			radiusSearchMode1(octreeLoc, point, radius, pointCloud, searchOctree);//内
			if (octreeLoc->level != octreeOG->level)
				radiusSearchMode2(octreeOG, octreeLoc, point, radius, pointCloud, searchOctree);//外
		}
		else {
			//Level < 1模式
			bool l1 = true;
			radiusSearchMode1(octreeLoc, point, radius, pointCloud, searchOctree, l1);//内
			if (octreeLoc->level != octreeOG->level)
				radiusSearchMode2(octreeOG, octreeLoc, point, radius, pointCloud, searchOctree, l1);//外
		}
	}
}

void Utility::radiusSearchMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree) {
	//for level > 1
	std::vector<std::vector<Octree::Octree_Struct>> vOctreeD2; //存储每一个level的octree向量集
	radiusSearchMode2(octreeLoc, vOctreeD2, point, radius);
	for (int k = 0; k < vOctreeD2.size(); k++) {
		for (int i = 0; i < vOctreeD2[k].size(); i++) {
			for (int j = 0; j < vOctreeD2[k][i]->points.size(); j++) {
				double dis = pow((point.x - vOctreeD2[k][i]->points[j].x), 2) + pow((point.y - vOctreeD2[k][i]->points[j].y), 2) + pow((point.z - vOctreeD2[k][i]->points[j].z), 2);
				if (dis <= pow(radius, 2)) {
					pointCloud.push_back(vOctreeD2[k][i]->points[j]);
					searchOctree.push_back(vOctreeD2[k][i]);
				}
			}
		}
	}
}

void Utility::radiusSearchMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree, bool l1) {
	//for level = 1
	std::vector<Octree::Octree_Struct> vOctree;
	getPointOctree(octreeLoc, vOctree);
	for (int i = 0; i < vOctree.size(); i++) {
		for (int j = 0; j < vOctree[i]->points.size(); j++) {
			double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
			if (dis <= pow(radius, 2)) {
				pointCloud.push_back(vOctree[i]->points[j]);
				searchOctree.push_back(vOctree[i]);
			}
		}
	}
}

void Utility::radiusSearchMode2(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius) {
	if (ostxyz->level == 1) {
		return ;
	}
	std::vector<Octree::Octree_Struct> vOctree;
	vOctree = getLowerOctree(ostxyz, point, radius);
	if (vOctree.size() > 0) {
		if (vOctree[0]->level == 1) {
			vOctreeD2.push_back(vOctree);
		}
	}
	for (int i = 0; i < vOctree.size(); i++)
		radiusSearchMode2(vOctree[i], vOctreeD2, point, radius);
}

void Utility::radiusSearchMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree) {
	//for level > 1
	Octree::Octree_Struct ostxyz;
	PointCloud::Point centerPoint;
	//XYZ系
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2)) {
		//(1,0,0)
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2)) {
		//(-1,0,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(0,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(0,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	//XOY面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(-1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(-1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	//XOZ面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	//YOZ面
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	//八个角
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchMode1(ostxyz, point, radius, pointCloud, searchOctree);
		}
	}
}

void Utility::radiusSearchMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, std::vector<Octree::Octree_Struct>& searchOctree, bool l1) {
	//for level = 1 代码精简后效率可能会低
	Octree::Octree_Struct ostxyz;
	PointCloud::Point centerPoint;
	//XYZ系
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2)) {
		//(1,0,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
			//std::cout << vOctree.size() << std::endl;
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2)) {
		//(-1,0,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
			//std::cout << ostxyz->center.x << std::endl;
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(0,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
			//std::cout << vOctree.size() << std::endl;
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(0,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
			//std::cout << vOctree.size() << std::endl;
		}
	}
	if ((point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
			//std::cout << vOctree.size() << std::endl;
		}
	}
	if ((point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
			//std::cout << vOctree.size() << std::endl;
		}
	}
	//XOY面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(-1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(-1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	//XOZ面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	//YOZ面
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	//八个角
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			getPointOctree(ostxyz, vOctree);
			for (int i = 0; i < vOctree.size(); i++) {
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					double dis = pow((point.x - vOctree[i]->points[j].x), 2) + pow((point.y - vOctree[i]->points[j].y), 2) + pow((point.z - vOctree[i]->points[j].z), 2);
					if (dis <= pow(radius, 2)) {
						pointCloud.push_back(vOctree[i]->points[j]);
						searchOctree.push_back(vOctree[i]);
					}
				}
			}
		}
	}
}

void Utility::radiusSearchCubeMode2(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius, int locLevel) {
	if (locLevel <= 6) {
		if (ostxyz->level == 1) {
			return;
		}
		std::vector<Octree::Octree_Struct> vOctree;
		vOctree = getLowerOctree(ostxyz, point, radius);
		if (vOctree.size() > 0) {
			if (vOctree[0]->level == 1) {
				vOctreeD2.push_back(vOctree);
			}
		}
		for (int i = 0; i < vOctree.size(); i++)
			radiusSearchCubeMode2(vOctree[i], vOctreeD2, point, radius, locLevel);
	}
	else {
		if (ostxyz->level == locLevel - 5) {
			return;
		}
		std::vector<Octree::Octree_Struct> vOctree;
		vOctree = getLowerOctree(ostxyz, point, radius);
		if (vOctree.size() > 0) {
			if (vOctree[0]->level == locLevel - 5) {
				vOctreeD2.push_back(vOctree);
			}
		}
		for (int i = 0; i < vOctree.size(); i++)
			radiusSearchCubeMode2(vOctree[i], vOctreeD2, point, radius, locLevel);
	}
}

void Utility::radiusSearchCubeModeByLevel(Octree::Octree_Struct ostxyz, std::vector<std::vector<Octree::Octree_Struct>>& vOctreeD2, PointCloud::Point point, double radius, int locLevel, int inputLevel) {
	if (locLevel <= 6 && inputLevel <= 6) {
		if (ostxyz->level == inputLevel) {
			return;
		}
		std::vector<Octree::Octree_Struct> vOctree;
		vOctree = getLowerOctree(ostxyz, point, radius);
		if (vOctree.size() > 0) {
			if (vOctree[0]->level == inputLevel) {
				vOctreeD2.push_back(vOctree);
			}
		}
		for (int i = 0; i < vOctree.size(); i++)
			radiusSearchCubeModeByLevel(vOctree[i], vOctreeD2, point, radius, locLevel, inputLevel);
	}
	else if(inputLevel <= locLevel){
		if (ostxyz->level == inputLevel) {
			return;
		}
		std::vector<Octree::Octree_Struct> vOctree;
		vOctree = getLowerOctree(ostxyz, point, radius);
		if (vOctree.size() > 0) {
			if (vOctree[0]->level == inputLevel) {
				vOctreeD2.push_back(vOctree);
			}
		}
		for (int i = 0; i < vOctree.size(); i++)
			radiusSearchCubeModeByLevel(vOctree[i], vOctreeD2, point, radius, locLevel, inputLevel);
	}
}

void Utility::radiusSearchCube(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud) {
	int locLevel = octreeLoc->level;
	//SearchMode1：pointXYZ + radius < octree.CenterXYZ 即可以以octree为顶层向下寻找点，寻找octree这个立方体内点
	if ((point.x + radius) < (octreeLoc->center.x + octreeLoc->length / 2) &&
		(point.x - radius) > (octreeLoc->center.x - octreeLoc->length / 2) &&
		(point.y + radius) < (octreeLoc->center.y + octreeLoc->length / 2) &&
		(point.y - radius) > (octreeLoc->center.y - octreeLoc->length / 2) &&
		(point.z + radius) < (octreeLoc->center.z + octreeLoc->length / 2) &&
		(point.z - radius) > (octreeLoc->center.z - octreeLoc->length / 2)) {
		radiusSearchCubeMode1(octreeLoc, point, radius, locLevel, pointCloud);
	}
	//SearchMode2：pointXYZ + radius在octree立方体外
	else {
		radiusSearchCubeMode1(octreeLoc, point, radius, locLevel, pointCloud);//内
		if (octreeLoc->level != octreeOG->level)
			radiusSearchCubeMode2(octreeOG, octreeLoc, point, radius, pointCloud);//外
	}
}

void Utility::radiusSearchCubeByLevel(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud, int inputLevel) {
	int locLevel = octreeLoc->level;
	//SearchMode1：pointXYZ + radius < octree.CenterXYZ 即可以以octree为顶层向下寻找点，寻找octree这个立方体内点
	if ((point.x + radius) < (octreeLoc->center.x + octreeLoc->length / 2) &&
		(point.x - radius) > (octreeLoc->center.x - octreeLoc->length / 2) &&
		(point.y + radius) < (octreeLoc->center.y + octreeLoc->length / 2) &&
		(point.y - radius) > (octreeLoc->center.y - octreeLoc->length / 2) &&
		(point.z + radius) < (octreeLoc->center.z + octreeLoc->length / 2) &&
		(point.z - radius) > (octreeLoc->center.z - octreeLoc->length / 2)) {
		radiusSearchCubeByLevelMode1(octreeLoc, point, radius, locLevel, inputLevel, pointCloud);
	}
	//SearchMode2：pointXYZ + radius在octree立方体外
	else {
		radiusSearchCubeByLevelMode1(octreeLoc, point, radius, locLevel, inputLevel, pointCloud);//内
		if (octreeLoc->level != octreeOG->level)
			radiusSearchCubeByLevelMode2(octreeOG, octreeLoc, point, radius, inputLevel, pointCloud);//外
	}
}

void Utility::radiusSearchCubeMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int locLevel, std::vector<PointCloud::Point>& pointCloud) {
	if (locLevel == 1) {
		std::vector<Octree::Octree_Struct> vOctree;
		getPointOctree(octreeLoc, vOctree);
		for (int i = 0; i < vOctree.size(); i++) {
			if (vOctree[i]->points.size() > 0) {
				PointCloud::Point ctrPoint = octreeLoc->center;
				if (vOctree[i]->points.size() > 0)
					ctrPoint.a = vOctree[i]->points[0].a;
				else
					ctrPoint.a = NO_ATTRIBUTE; //未定义合适属性
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					if (ctrPoint.a != vOctree[i]->points[j].a) {
						ctrPoint.a = MIX_ATTRIBUTE;
						j = vOctree[i]->points.size();
					}
				}
				pointCloud.push_back(ctrPoint);
				return;
			}
		}
		return;
	}

	std::vector<std::vector<Octree::Octree_Struct>> vOctreeD2; //存储每一个level的octree向量集
	radiusSearchCubeMode2(octreeLoc, vOctreeD2, point, radius, locLevel);
	for (int k = 0; k < vOctreeD2.size(); k++) {
		for (int i = 0; i < vOctreeD2[k].size(); i++) {
			std::vector<Octree::Octree_Struct> vOctree;
			getPointOctree(vOctreeD2[k][i], vOctree);
			for (int j = 0; j < vOctree.size(); j++) {
				if (vOctree[j]->points.size() > 0) {
					PointCloud::Point ctrPoint = vOctreeD2[k][i]->center;
					ctrPoint.a = vOctree[j]->points[0].a;
					for (int m = 0; m < vOctree.size(); m++) {
						for (int n = 0; n < vOctree[m]->points.size(); n++) {
							if (ctrPoint.a != vOctree[m]->points[n].a) {
								ctrPoint.a = MIX_ATTRIBUTE;
								n = vOctree[m]->points.size();
								//NVCC bug?  m = vOctree.size();
							}
						}
						if (ctrPoint.a == MIX_ATTRIBUTE)
							m = vOctree.size();
					}
					pointCloud.push_back(ctrPoint);
					j += vOctree.size();
				}
			}
		}
	}
}

void Utility::radiusSearchCubeByLevelMode1(Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int locLevel, int inputLevel, std::vector<PointCloud::Point>& pointCloud) {
	if (locLevel == 1) {
		std::vector<Octree::Octree_Struct> vOctree;
		getPointOctree(octreeLoc, vOctree);
		for (int i = 0; i < vOctree.size(); i++) {
			if (vOctree[i]->points.size() > 0) {
				PointCloud::Point ctrPoint = octreeLoc->center;
				if (vOctree[i]->points.size() > 0)
					ctrPoint.a = vOctree[i]->points[0].a;
				else
					ctrPoint.a = NO_ATTRIBUTE; //未定义合适属性
				for (int j = 0; j < vOctree[i]->points.size(); j++) {
					if (ctrPoint.a != vOctree[i]->points[j].a) {
						ctrPoint.a = MIX_ATTRIBUTE;
						j = vOctree[i]->points.size();
					}
				}
				pointCloud.push_back(ctrPoint);
				return;
			}
		}
		return;
	}
	std::vector<std::vector<Octree::Octree_Struct>> vOctreeD2; //存储每一个level的octree向量集
	radiusSearchCubeModeByLevel(octreeLoc, vOctreeD2, point, radius, locLevel, inputLevel);
	for (int k = 0; k < vOctreeD2.size(); k++) {
		for (int i = 0; i < vOctreeD2[k].size(); i++) {
			std::vector<Octree::Octree_Struct> vOctree;
			getPointOctree(vOctreeD2[k][i], vOctree);
			for (int j = 0; j < vOctree.size(); j++) {
				if (vOctree[j]->points.size() > 0) {
					PointCloud::Point ctrPoint = vOctreeD2[k][i]->center;
					ctrPoint.a = vOctree[j]->points[0].a;
					for (int m = 0; m < vOctree.size(); m++) {
						for (int n = 0; n < vOctree[m]->points.size(); n++) {
							if (ctrPoint.a != vOctree[m]->points[n].a) {
								ctrPoint.a = MIX_ATTRIBUTE;
								n = vOctree[m]->points.size();
								//NVCC bug?  m = vOctree.size();
							}
						}
						if (ctrPoint.a == MIX_ATTRIBUTE)
							m = vOctree.size();
					}
					pointCloud.push_back(ctrPoint);
					j += vOctree.size();
				}
			}
		}
	}
}

void Utility::radiusSearchCubeMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, std::vector<PointCloud::Point>& pointCloud) {
	Octree::Octree_Struct ostxyz;
	PointCloud::Point centerPoint;
	int locLevel = octreeLoc->level;
	//XYZ系
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2)) {
		//(1,0,0)
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2)) {
		//(-1,0,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(0,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(0,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	//XOY面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(-1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(-1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	//XOZ面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	//YOZ面
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	//八个角
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeMode1(ostxyz, point, radius, locLevel, pointCloud);
		}
	}
}

void Utility::radiusSearchCubeByLevelMode2(Octree::Octree_Struct octreeOG, Octree::Octree_Struct octreeLoc, PointCloud::Point point, double radius, int inputLevel, std::vector<PointCloud::Point>& pointCloud) {
	Octree::Octree_Struct ostxyz;
	PointCloud::Point centerPoint;
	int locLevel = octreeLoc->level;
	//XYZ系
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2)) {
		//(1,0,0)
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2)) {
		//(-1,0,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.x == centerPoint.x) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(0,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(0,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.y == centerPoint.y) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if (ostxyz->center.z == centerPoint.z) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	//XOY面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2)) {
		//(-1,1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2)) {
		//(-1,-1,0)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	//XOZ面
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,0,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	//YOZ面
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(0,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(0,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	//八个角
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(-1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,-1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x - radius) <= (octreeLoc->center.x - octreeLoc->length / 2) && (point.y + radius) >= (octreeLoc->center.y + octreeLoc->length / 2) && (point.z - radius) <= (octreeLoc->center.z - octreeLoc->length / 2)) {
		//(-1,1,-1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x - octreeLoc->length, octreeLoc->center.y + octreeLoc->length, octreeLoc->center.z - octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
	if ((point.x + radius) >= (octreeLoc->center.x + octreeLoc->length / 2) && (point.y - radius) <= (octreeLoc->center.y - octreeLoc->length / 2) && (point.z + radius) >= (octreeLoc->center.z + octreeLoc->length / 2)) {
		//(1,-1,1)
		std::vector<Octree::Octree_Struct> vOctree;
		centerPoint = PointCloud::Point(octreeLoc->center.x + octreeLoc->length, octreeLoc->center.y - octreeLoc->length, octreeLoc->center.z + octreeLoc->length);
		ostxyz = locatePoints(octreeOG, centerPoint, octreeLoc->level);
		if ((ostxyz->center.x == centerPoint.x) && (ostxyz->center.y == centerPoint.y) && (ostxyz->center.z == centerPoint.z)) {
			radiusSearchCubeByLevelMode1(ostxyz, point, radius, locLevel, inputLevel, pointCloud);
		}
	}
}