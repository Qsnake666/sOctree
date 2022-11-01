#include "Octree.cuh"
#define THREADS_PER_BLOCK 512
#define cudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s, %s, %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void maxValueVectorKernel(double* A, double* B, unsigned int size) {
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int i = bid * blockDim.x + tid;

	/* find the maximum value of each block using a reduction */
	if (i < size) {
		unsigned int stride;
		for (stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
			__shared__ double As[THREADS_PER_BLOCK];
			if (tid < stride) {
				As[tid] = A[i];
				As[tid + stride] = A[i + stride];
				if (As[tid] < As[tid + stride])
					A[i] = As[tid + stride];
			}
		}
	}
	__syncthreads();

	if (tid == 0)
		B[bid] = A[0 + bid * blockDim.x];
}

__global__ void minValueVectorKernel(double* A, double* B, unsigned int size) {
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int i = bid * blockDim.x + tid;

	/* find the maximum value of each block using a reduction */
	if (i < size) {
		unsigned int stride;
		for (stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
			__shared__ double As[THREADS_PER_BLOCK];
			if (tid < stride) {
				As[tid] = A[i];
				As[tid + stride] = A[i + stride];
				if (As[tid] > As[tid + stride])
					A[i] = As[tid + stride];
			}
		}
	}
	__syncthreads();

	if (tid == 0)
		B[bid] = A[0 + bid * blockDim.x];
}

void Octree::Octree_Node::init(Octree_Node* parent, double length, int level) {
	this->parent = parent;
	for (int i = 0; i < 8; i++) {
		nodes[i] = NULL;
	}
	this->level = level;
	this->length = length;
}

void Octree::Octree_Node::destory() {
	this->parent = NULL;
	for (int i = 0; i < 8; i++) {
		nodes[i] = NULL;
	}
}

void  Octree::setPoint(std::vector<PointCloud::Point>& points) {
	pointCloud = points;
	//向量的指针如果要访问向量数据类型内容需要实例化一个向量
	//无法直接访问
}

void Octree::maxValue(std::vector<PointCloud::Point> pointCloud, double& overall_max_x, double& overall_max_y, double& overall_max_z)
{
	unsigned int size, size2;
	unsigned int i;  // loop index
	double* xdata;
	double* ydata;
	double* zdata;

	size2 = pointCloud.capacity();
	size = (size2 / THREADS_PER_BLOCK) * THREADS_PER_BLOCK + THREADS_PER_BLOCK;

	int NUM_BLOCKS;
	NUM_BLOCKS = size / THREADS_PER_BLOCK;
	/*
	std::vector<Point> pointCloud;
	for (int i = 0; i < size2; i++) {
		//pointCloud.push_back(Point((long long)rand() / ((double)(RAND_MAX) / 100) * -1, (long long)rand() / ((double)(RAND_MAX) / 100), (long long)rand() / ((double)(RAND_MAX) / 100)));
		pointCloud.push_back(Point(-i - 1, i - 11, i + 1));
	}
	*/
	/*效率测试
	double max_x, max_y, max_z;
	max_x = pointCloud[0].x; max_y = pointCloud[0].y; max_z = pointCloud[0].z;
	std::vector<Point>::iterator it = pointCloud.begin(); //遍历整个vector
	for (; it != pointCloud.end(); it++) {
		//找到最大的x，y，z值
		max_x = max_x < (*it).x ? (*it).x : max_x;
		max_y = max_y < (*it).y ? (*it).y : max_y;
		max_z = max_z < (*it).z ? (*it).z : max_z;
	}
	*/

	xdata = (double*)malloc(size * sizeof(double));
	ydata = (double*)malloc(size * sizeof(double));
	zdata = (double*)malloc(size * sizeof(double));

	for (i = 0; i < size2; i++) {
		xdata[i] = pointCloud[i].x;
		ydata[i] = pointCloud[i].y;
		zdata[i] = pointCloud[i].z;
	}

	//补齐操作
	for (i = size2; i < size; i++) {
		xdata[i] = pointCloud[0].x;
		ydata[i] = pointCloud[0].y;
		zdata[i] = pointCloud[0].z;
	}

	double* maxes_x = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* dev_num_x, * dev_maxes_x;
	double* maxes_y = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* dev_num_y, * dev_maxes_y;
	double* maxes_z = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* dev_num_z, * dev_maxes_z;

	cudaError(cudaMalloc((void**)&dev_num_x, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_maxes_x, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_num_y, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_maxes_y, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_num_z, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_maxes_z, NUM_BLOCKS * sizeof(double)));

	cudaStream_t streams[3];
	for (int i = 0; i < 3; i++)
	{
		cudaStreamCreate(&(streams[i]));
	}

	cudaError(cudaMemcpyAsync(dev_num_x, xdata, size * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
	cudaError(cudaMemcpyAsync(dev_num_y, ydata, size * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
	cudaError(cudaMemcpyAsync(dev_num_z, zdata, size * sizeof(double), cudaMemcpyHostToDevice, streams[2]));

	//并发
	maxValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[0] >> > (dev_num_x, dev_maxes_x, size);
	maxValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[1] >> > (dev_num_y, dev_maxes_y, size);
	maxValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[2] >> > (dev_num_z, dev_maxes_z, size);

	cudaError(cudaPeekAtLastError()); //debug info
	cudaError(cudaDeviceSynchronize());

	cudaError(cudaMemcpyAsync(maxes_x, dev_maxes_x, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
	cudaError(cudaMemcpyAsync(maxes_y, dev_maxes_y, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[1]));
	cudaError(cudaMemcpyAsync(maxes_z, dev_maxes_z, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[2]));

	cudaDeviceSynchronize();
	cudaFree(dev_num_x);
	cudaFree(dev_maxes_x);
	free(xdata);
	cudaFree(dev_num_y);
	cudaFree(dev_maxes_y);
	free(ydata);
	cudaFree(dev_num_z);
	cudaFree(dev_maxes_z);
	free(zdata);

	size2 = NUM_BLOCKS;
	size = size2 / THREADS_PER_BLOCK * THREADS_PER_BLOCK + THREADS_PER_BLOCK;
	NUM_BLOCKS = size / THREADS_PER_BLOCK;

	double* maxestmp_x = (double*)malloc(size * sizeof(double));
	double* maxestmp_y = (double*)malloc(size * sizeof(double));
	double* maxestmp_z = (double*)malloc(size * sizeof(double));

	//补齐
	for (i = 0; i < size2; i++) {
		maxestmp_x[i] = maxes_x[i];
		maxestmp_y[i] = maxes_y[i];
		maxestmp_z[i] = maxes_z[i];
	}
	for (i = size2; i < size; i++) {
		maxestmp_x[i] = maxes_x[0];
		maxestmp_y[i] = maxes_y[0];
		maxestmp_z[i] = maxes_z[0];
	}


	double* maxes2_x = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* maxes2_y = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* maxes2_z = (double*)malloc(NUM_BLOCKS * sizeof(double));

	cudaError(cudaMalloc((void**)&dev_num_x, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_maxes_x, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMemcpyAsync(dev_num_x, maxestmp_x, size * sizeof(double), cudaMemcpyHostToDevice, streams[0]));

	cudaError(cudaMalloc((void**)&dev_num_y, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_maxes_y, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMemcpyAsync(dev_num_y, maxestmp_y, size * sizeof(double), cudaMemcpyHostToDevice, streams[1]));

	cudaError(cudaMalloc((void**)&dev_num_z, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_maxes_z, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMemcpyAsync(dev_num_z, maxestmp_z, size * sizeof(double), cudaMemcpyHostToDevice, streams[2]));

	maxValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[0] >> > (dev_num_x, dev_maxes_x, size);
	maxValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[1] >> > (dev_num_y, dev_maxes_y, size);
	maxValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[2] >> > (dev_num_z, dev_maxes_z, size);

	cudaError(cudaPeekAtLastError()); //debug info
	cudaError(cudaDeviceSynchronize());

	cudaError(cudaMemcpyAsync(maxes2_x, dev_maxes_x, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
	cudaError(cudaMemcpyAsync(maxes2_y, dev_maxes_y, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[1]));
	cudaError(cudaMemcpyAsync(maxes2_z, dev_maxes_z, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[2]));

	overall_max_x = maxes2_x[0];
	overall_max_y = maxes2_y[0];
	overall_max_z = maxes2_z[0];

	for (i = 1; i < NUM_BLOCKS; ++i) {
		if (overall_max_x < maxes2_x[i])
			overall_max_x = maxes2_x[i];
	}
	//printf(" The maximum number in the arrayX is: %f\n", overall_max_x);

	for (i = 1; i < NUM_BLOCKS; ++i) {
		if (overall_max_y < maxes2_y[i])
			overall_max_y = maxes2_y[i];
	}
	//printf(" The maximum number in the arrayY is: %f\n", overall_max_y);

	for (i = 1; i < NUM_BLOCKS; ++i) {
		if (overall_max_z < maxes2_z[i])
			overall_max_z = maxes2_z[i];
	}
	//printf(" The maximum number in the arrayZ is: %f\n", overall_max_z);

	cudaDeviceSynchronize();
	cudaFree(dev_num_x);
	cudaFree(dev_maxes_x);
	free(maxes_x);
	cudaFree(dev_num_y);
	cudaFree(dev_maxes_y);
	free(maxes_y);
	cudaFree(dev_num_z);
	cudaFree(dev_maxes_z);
	free(maxes_z);

	for (int i = 0; i < 3; ++i)
		cudaStreamDestroy(streams[i]);
}

void Octree::minValue(std::vector<PointCloud::Point> pointCloud, double& overall_min_x, double& overall_min_y, double& overall_min_z)
{
	unsigned int size, size2;
	unsigned int i;  // loop index
	double* xdata;
	double* ydata;
	double* zdata;

	size2 = pointCloud.capacity();
	size = (size2 / THREADS_PER_BLOCK) * THREADS_PER_BLOCK + THREADS_PER_BLOCK;
	int NUM_BLOCKS;
	NUM_BLOCKS = size / THREADS_PER_BLOCK;



	/*  //效率测试
	double min_x, min_y, min_z;
	min_x = pointCloud[0].x; min_y = pointCloud[0].y; min_z = pointCloud[0].z;
	std::vector<Point>::iterator it = pointCloud.begin(); //遍历整个vector
	for (; it != pointCloud.end(); it++) {
		//找到最大的x，y，z值
		min_x = min_x > (*it).x ? (*it).x : min_x;
		min_y = min_y > (*it).y ? (*it).y : min_y;
		min_z = min_z > (*it).z ? (*it).z : min_z;
	}
	*/

	xdata = (double*)malloc(size * sizeof(double));
	ydata = (double*)malloc(size * sizeof(double));
	zdata = (double*)malloc(size * sizeof(double));

	for (i = 0; i < size2; i++) {
		xdata[i] = pointCloud[i].x;
		ydata[i] = pointCloud[i].y;
		zdata[i] = pointCloud[i].z;
	}

	//补齐操作
	for (i = size2; i < size; i++) {
		xdata[i] = pointCloud[0].x;
		ydata[i] = pointCloud[0].y;
		zdata[i] = pointCloud[0].z;
	}

	double* mins_x = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* dev_num_x, * dev_mins_x;
	double* mins_y = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* dev_num_y, * dev_mins_y;
	double* mins_z = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* dev_num_z, * dev_mins_z;

	cudaError(cudaMalloc((void**)&dev_num_x, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_mins_x, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_num_y, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_mins_y, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_num_z, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_mins_z, NUM_BLOCKS * sizeof(double)));

	cudaStream_t streams[3];
	for (int i = 0; i < 3; i++)
	{
		cudaStreamCreate(&(streams[i]));
	}

	cudaError(cudaMemcpyAsync(dev_num_x, xdata, size * sizeof(double), cudaMemcpyHostToDevice, streams[0]));
	cudaError(cudaMemcpyAsync(dev_num_y, ydata, size * sizeof(double), cudaMemcpyHostToDevice, streams[1]));
	cudaError(cudaMemcpyAsync(dev_num_z, zdata, size * sizeof(double), cudaMemcpyHostToDevice, streams[2]));

	minValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[0] >> > (dev_num_x, dev_mins_x, size);
	minValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[1] >> > (dev_num_y, dev_mins_y, size);
	minValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[2] >> > (dev_num_z, dev_mins_z, size);

	cudaError(cudaPeekAtLastError()); //debug info
	cudaError(cudaDeviceSynchronize());

	cudaError(cudaMemcpyAsync(mins_x, dev_mins_x, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
	cudaError(cudaMemcpyAsync(mins_y, dev_mins_y, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[1]));
	cudaError(cudaMemcpyAsync(mins_z, dev_mins_z, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[2]));

	cudaDeviceSynchronize();
	cudaFree(dev_num_x);
	cudaFree(dev_mins_x);
	free(xdata);
	cudaFree(dev_num_y);
	cudaFree(dev_mins_y);
	free(ydata);
	cudaFree(dev_num_z);
	cudaFree(dev_mins_z);
	free(zdata);

	size2 = NUM_BLOCKS;
	size = size2 / THREADS_PER_BLOCK * THREADS_PER_BLOCK + THREADS_PER_BLOCK;

	double* minstmp_x = (double*)malloc(size * sizeof(double));
	double* minstmp_y = (double*)malloc(size * sizeof(double));
	double* minstmp_z = (double*)malloc(size * sizeof(double));

	//补齐
	for (i = 0; i < size2; i++) {
		minstmp_x[i] = mins_x[i];
		minstmp_y[i] = mins_y[i];
		minstmp_z[i] = mins_z[i];
	}
	for (i = size2; i < size; i++) {
		minstmp_x[i] = mins_x[0];
		minstmp_y[i] = mins_y[0];
		minstmp_z[i] = mins_z[0];
	}

	NUM_BLOCKS = size / THREADS_PER_BLOCK;

	double* mins2_x = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* mins2_y = (double*)malloc(NUM_BLOCKS * sizeof(double));
	double* mins2_z = (double*)malloc(NUM_BLOCKS * sizeof(double));

	cudaError(cudaMalloc((void**)&dev_num_x, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_mins_x, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMemcpyAsync(dev_num_x, minstmp_x, size * sizeof(double), cudaMemcpyHostToDevice, streams[0]));

	cudaError(cudaMalloc((void**)&dev_num_y, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_mins_y, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMemcpyAsync(dev_num_y, minstmp_y, size * sizeof(double), cudaMemcpyHostToDevice, streams[1]));

	cudaError(cudaMalloc((void**)&dev_num_z, size * sizeof(double)));
	cudaError(cudaMalloc((void**)&dev_mins_z, NUM_BLOCKS * sizeof(double)));
	cudaError(cudaMemcpyAsync(dev_num_z, minstmp_z, size * sizeof(double), cudaMemcpyHostToDevice, streams[2]));

	minValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[0] >> > (dev_num_x, dev_mins_x, size);
	minValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[1] >> > (dev_num_y, dev_mins_y, size);
	minValueVectorKernel << <NUM_BLOCKS, THREADS_PER_BLOCK, 0, streams[2] >> > (dev_num_z, dev_mins_z, size);

	cudaError(cudaPeekAtLastError()); //debug info
	cudaError(cudaDeviceSynchronize());

	cudaError(cudaMemcpyAsync(mins2_x, dev_mins_x, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[0]));
	cudaError(cudaMemcpyAsync(mins2_y, dev_mins_y, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[1]));
	cudaError(cudaMemcpyAsync(mins2_z, dev_mins_z, NUM_BLOCKS * sizeof(double), cudaMemcpyDeviceToHost, streams[2]));

	overall_min_x = mins2_x[0];
	overall_min_y = mins2_y[0];
	overall_min_z = mins2_z[0];
	//为了防止数字零的干扰

	for (i = 0; i < NUM_BLOCKS; ++i) {
		if (overall_min_x > mins2_x[i])
			overall_min_x = mins2_x[i];
	}
	//printf(" The minimum number in the arrayX is: %f\n", overall_min_x);

	for (i = 0; i < NUM_BLOCKS; ++i) {
		if (overall_min_y > mins2_y[i])
			overall_min_y = mins2_y[i];
	}
	//printf(" The minimum number in the arrayY is: %f\n", overall_min_y);

	for (i = 0; i < NUM_BLOCKS; ++i) {
		if (overall_min_z > mins2_z[i])
			overall_min_z = mins2_z[i];
	}
	//printf(" The minimum number in the arrayZ is: %f\n", overall_min_z);

	cudaDeviceSynchronize();
	cudaFree(dev_num_x);
	cudaFree(dev_mins_x);
	free(mins_x);
	cudaFree(dev_num_y);
	cudaFree(dev_mins_y);
	free(mins_y);
	cudaFree(dev_num_z);
	cudaFree(dev_mins_z);
	free(mins_z);

	for (int i = 0; i < 3; ++i)
		cudaStreamDestroy(streams[i]);
}

void Octree::CreatOctreeByPointCloud() {
	/*
	this->max_x = pointCloud.at(0).x; //指向输入点云第一个点
	this->max_y = pointCloud.at(0).y; //只是初始化
	this->max_z = pointCloud.at(0).z;
	this->min_x = pointCloud.at(0).x;
	this->min_y = pointCloud.at(0).y;
	this->min_z = pointCloud.at(0).z;

	//计算八叉树深度和宽度 CUDA：已完成
	std::vector<Point>::iterator it = pointCloud.begin(); //遍历整个vector
	for (; it != pointCloud.end(); it++) {
		//找到最大的x，y，z值
		this->max_x = this->max_x < (*it).x ? (*it).x : this->max_x;
		this->max_y = this->max_y < (*it).y ? (*it).y : this->max_y;
		this->max_z = this->max_z < (*it).z ? (*it).z : this->max_z;
		this->min_x = this->min_x > (*it).x ? (*it).x : this->min_x;
		this->min_y = this->min_y > (*it).y ? (*it).y : this->min_y;
		this->min_z = this->min_z > (*it).z ? (*it).z : this->min_z;
	}
	*/
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

	while (length < maxLength) {
		length *= 2;	//但在八叉树，每层长度都是子节点的两倍
		level++;	//所以根节点的length要大于或等于maxLength
	} //最终得到根节点（最大正方体）length和level

	//初始化八叉树
	//根节点octree
	//length：根节点每个格子的宽度，格子为正方体
	//level：将要创建的八叉树的深度最大值，最小length对应level=1
	//	比如深度为5的八叉树，根节点的level就是5，叶子节点1
	octree = new Octree_Node();
	//octLength为手动设置叶子节点格子大小，根据点云规模设置，越小八叉树分的越细
	octree->init(NULL, length, level);	//初始化根节点（最大正方体）（该函数Step1）
	double cx = (max_x + min_x) * 0.5;
	double cy = (max_y + min_y) * 0.5;
	double cz = (max_z + min_z) * 0.5;
	//将整个点云中心设置为根节点中心
	octree->center.setPoint(cx, cy, cz);
	//生成一个空的八叉树，不包含任何数据
	creat(octree);
	//把数据缩放到八叉树中，并把不包含数据的地方释放掉
	addPointCloud(pointCloud);
}

void Octree::creat(Octree_Struct octree) {
	if (octree->level == 1) {
		return;
	}
	for (int i = 0; i < 8; i++) {
		octree->nodes[i] = new Octree_Node();
		octree->nodes[i]->init(octree, octree->length / 2, octree->level - 1);

		if (i == 0 || i == 2 || i == 4 || i == 6)
			octree->nodes[i]->center.x = octree->center.x + octree->length / 4;
		if (i == 1 || i == 3 || i == 5 || i == 7)
			octree->nodes[i]->center.x = octree->center.x - octree->length / 4;
		if (i == 0 || i == 1 || i == 4 || i == 5)
			octree->nodes[i]->center.y = octree->center.y + octree->length / 4;
		if (i == 2 || i == 3 || i == 6 || i == 7)
			octree->nodes[i]->center.y = octree->center.y - octree->length / 4;
		if (i == 0 || i == 1 || i == 2 || i == 3)
			octree->nodes[i]->center.z = octree->center.z + octree->length / 4;
		if (i == 4 || i == 5 || i == 6 || i == 7)
			octree->nodes[i]->center.z = octree->center.z - octree->length / 4;

		creat(octree->nodes[i]);
	}
}

void Octree::addPointCloud(std::vector<PointCloud::Point> pointCloud) {
	std::vector<PointCloud::Point>::iterator it = pointCloud.begin();
	for (; it != pointCloud.end(); it++) {
		octree->count++;
		addNode(octree, *it);
	}
	//addNodeEnd(octree); //不释放空节点防止radiusSearch空指针报错
}

void Octree::addNode(Octree_Struct octree, PointCloud::Point point) {
	if (octree->level == 1) {
		octree->points.push_back(point);
		return;
	}
	if (point.x >= octree->center.x && point.y >= octree->center.y && point.z >= octree->center.z) {
		octree->nodes[0]->count++;
		addNode(octree->nodes[0], point);
	}
	else if (point.x < octree->center.x && point.y >= octree->center.y && point.z >= octree->center.z) {
		octree->nodes[1]->count++;
		addNode(octree->nodes[1], point);
	}
	else if (point.x >= octree->center.x && point.y < octree->center.y && point.z >= octree->center.z) {
		octree->nodes[2]->count++;
		addNode(octree->nodes[2], point);
	}
	else if (point.x < octree->center.x && point.y < octree->center.y && point.z >= octree->center.z) {
		octree->nodes[3]->count++;
		addNode(octree->nodes[3], point);
	}
	else if (point.x >= octree->center.x && point.y >= octree->center.y && point.z < octree->center.z) {
		octree->nodes[4]->count++;
		addNode(octree->nodes[4], point);
	}
	else if (point.x < octree->center.x && point.y >= octree->center.y && point.z < octree->center.z) {
		octree->nodes[5]->count++;
		addNode(octree->nodes[5], point);
	}
	else if (point.x >= octree->center.x && point.y < octree->center.y && point.z < octree->center.z) {
		octree->nodes[6]->count++;
		addNode(octree->nodes[6], point);
	}
	else if (point.x < octree->center.x && point.y < octree->center.y && point.z < octree->center.z) {
		octree->nodes[7]->count++;
		addNode(octree->nodes[7], point);
	}
}

void Octree::addNodeEnd(Octree_Struct octree) {
	for (int i = 0; i < 8; i++) {
		if (octree->nodes[i] != NULL) {
			addNodeEnd(octree->nodes[i]);
			if (octree->nodes[i]->count == 0) {
				octree->nodes[i]->parent = NULL;
				delete octree->nodes[i];
				octree->nodes[i] = NULL;
			}
		}
	}
}

void Octree::destory(Octree_Struct octree) {
	for (int i = 0; i < 8; i++) {
		if (octree->nodes[i] != NULL) {
			destory(octree->nodes[i]);
			octree->nodes[i]->parent = NULL;
			octree->nodes[i]->points.clear();
			delete octree->nodes[i];
		}
	}
	if (octree->parent == NULL) {
		pointCloud.clear();
		delete octree;
	}
}

