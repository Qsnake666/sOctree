#include "Visualization.cuh"
#define BLOCK_SIZE  1

__device__ void SetElement(Visualization::Matrix A, int row, int col, double value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Visualization::Matrix GetSubMatrix(Visualization::Matrix A, int row, int col) {
    Visualization::Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];//A[BLOCK_SZIE*row][BLOCK_SIZE*col]
    return Asub;
}

__global__ void MatMulKernel(Visualization::Matrix A, Visualization::Matrix B, Visualization::Matrix C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    double Cvalue = 0;
    int row = threadIdx.y;
    int col = threadIdx.x;
    //矩阵A的子矩阵开始的在A矩阵的index
    int aBegin = A.width * BLOCK_SIZE * blockRow;//A[BLOCK_DIM*by][0]

    //矩阵A的子矩阵结束的在A矩阵的index
    int aEnd = aBegin + A.width - 1;

    //Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;//每一次加A的一个block的列数

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockCol; //B[0][BLOCK_DIM*bx]

    //Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * B.width;//每一次加B的一个block行数

    /*
    if Rotate
    if (ty < 3 && tx < 4)
        shared_mul[ty][tx] = mul[ty][tx];
    }
    */

    //共享内存一个bx值计算A一行，一个by值计算B一列
    for (int a = aBegin, b = bBegin;
        a <= aEnd;
        a += aStep, b += bStep)
    {
        /*for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            Matrix Bsub = GetSubMatrix(B, m, blockCol);*/
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
        /*As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        __syncthreads();*/
        As[row][col] = A.elements[a + A.width * row + col]; //As[16*by+ty][tx] +step As[16*by+ty][tx+BLOCK_DIM] 横向延申
        Bs[row][col] = B.elements[b + B.width * row + col];
        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col]; //这里按照列循环，一列就一个Cvalue
        __syncthreads();
    }
    SetElement(C, blockRow * BLOCK_SIZE + row, blockCol * BLOCK_SIZE + col, Cvalue);
}

__global__ void MatTranKernel(Visualization::Matrix A, Visualization::Matrix B) {
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int absy = by * BLOCK_SIZE + row;
    int absx = bx * BLOCK_SIZE + col;

    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    As[row][col] = A.elements[absy * A.width + absx];
    __syncthreads();

    SetElement(B, bx * BLOCK_SIZE + col, by * BLOCK_SIZE + row, As[row][col]);
}

void Visualization::vectorToMatrix(std::vector<PointCloud::Point> pointCloud, Matrix* A) {
    A->width = 4;//添加属性数据已经更改！
    A->height = pointCloud.size();
    A->stride = A->width;
    A->elements = (double*)calloc(A->height * A->width, sizeof(double));
    for (int i = 0; i < A->height; i++) {
        A->elements[i * A->stride + 0] = pointCloud[i].x;
        A->elements[i * A->stride + 1] = pointCloud[i].y;
        A->elements[i * A->stride + 2] = pointCloud[i].z;
        A->elements[i * A->stride + 3] = pointCloud[i].a;
    }
}

void Visualization::vectorToMatrixCube(std::vector<PointCloud::Point> pointCloud, Matrix* A, double length) {
    A->width = 4;//添加属性数据已经更改！
    A->height = pointCloud.size() * 9;//添加8方向
    A->stride = A->width;
    A->elements = (double*)calloc(A->height * A->width, sizeof(double));
    int idx = 0;
    for (int i = 0; i < A->height; i += 9) {
        //中心点
        A->elements[i * A->stride + 0] = pointCloud[idx].x;
        A->elements[i * A->stride + 1] = pointCloud[idx].y;
        A->elements[i * A->stride + 2] = pointCloud[idx].z;
        A->elements[i * A->stride + 3] = pointCloud[idx].a;
        //(1,1,1)
        A->elements[(i + 1) * A->stride + 0] = pointCloud[idx].x + length / 2;
        A->elements[(i + 1) * A->stride + 1] = pointCloud[idx].y + length / 2;
        A->elements[(i + 1) * A->stride + 2] = pointCloud[idx].z + length / 2;
        A->elements[(i + 1) * A->stride + 3] = pointCloud[idx].a;
        //(-1,1,1)
        A->elements[(i + 2) * A->stride + 0] = pointCloud[idx].x - length / 2;
        A->elements[(i + 2) * A->stride + 1] = pointCloud[idx].y + length / 2;
        A->elements[(i + 2) * A->stride + 2] = pointCloud[idx].z + length / 2;
        A->elements[(i + 2) * A->stride + 3] = pointCloud[idx].a;
        //(-1,-1,1)
        A->elements[(i + 3) * A->stride + 0] = pointCloud[idx].x - length / 2;
        A->elements[(i + 3) * A->stride + 1] = pointCloud[idx].y - length / 2;
        A->elements[(i + 3) * A->stride + 2] = pointCloud[idx].z + length / 2;
        A->elements[(i + 3) * A->stride + 3] = pointCloud[idx].a;
        //(1,-1,1)
        A->elements[(i + 4) * A->stride + 0] = pointCloud[idx].x + length / 2;
        A->elements[(i + 4) * A->stride + 1] = pointCloud[idx].y - length / 2;
        A->elements[(i + 4) * A->stride + 2] = pointCloud[idx].z + length / 2;
        A->elements[(i + 4) * A->stride + 3] = pointCloud[idx].a;

        //(1,1,-1)
        A->elements[(i + 5) * A->stride + 0] = pointCloud[idx].x + length / 2;
        A->elements[(i + 5) * A->stride + 1] = pointCloud[idx].y + length / 2;
        A->elements[(i + 5) * A->stride + 2] = pointCloud[idx].z - length / 2;
        A->elements[(i + 5) * A->stride + 3] = pointCloud[idx].a;
        //(-1,1,-1)
        A->elements[(i + 6) * A->stride + 0] = pointCloud[idx].x - length / 2;
        A->elements[(i + 6) * A->stride + 1] = pointCloud[idx].y + length / 2;
        A->elements[(i + 6) * A->stride + 2] = pointCloud[idx].z - length / 2;
        A->elements[(i + 6) * A->stride + 3] = pointCloud[idx].a;
        //(-1,-1,-1)
        A->elements[(i + 7) * A->stride + 0] = pointCloud[idx].x - length / 2;
        A->elements[(i + 7) * A->stride + 1] = pointCloud[idx].y - length / 2;
        A->elements[(i + 7) * A->stride + 2] = pointCloud[idx].z - length / 2;
        A->elements[(i + 7) * A->stride + 3] = pointCloud[idx].a;
        //(1,-1,-1)
        A->elements[(i + 8) * A->stride + 0] = pointCloud[idx].x + length / 2;
        A->elements[(i + 8) * A->stride + 1] = pointCloud[idx].y - length / 2;
        A->elements[(i + 8) * A->stride + 2] = pointCloud[idx].z - length / 2;
        A->elements[(i + 8) * A->stride + 3] = pointCloud[idx].a;

        idx += 1;
    }
}

void Visualization::matrixMul(Matrix A, Matrix B, Matrix C) {
    Matrix d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc((void**)&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(double);
    cudaMalloc((void**)&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
        cudaMemcpyHostToDevice);

    Matrix d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(double);
    cudaMalloc((void**)&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);
    cudaMemcpy(C.elements, d_C.elements, size,
        cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

void Visualization::matrixTranspose(Matrix A, Matrix B) {
    Matrix d_A;
    d_A.width = A.width;
    d_A.height = A.height;
    d_A.stride = A.width;
    size_t size = A.width * A.height * sizeof(double);
    cudaMalloc((void**)&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
        cudaMemcpyHostToDevice);

    Matrix d_B;
    d_B.width = B.width;
    d_B.height = B.height;
    d_B.stride = B.width;
    size = B.width * B.height * sizeof(double);
    cudaMalloc((void**)&d_B.elements, size);


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y);

    MatTranKernel << <dimGrid, dimBlock >> > (d_A, d_B);
    cudaMemcpy(B.elements, d_B.elements, size,
        cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
}

void Visualization::initMatrix(Visualization::Matrix* A, int row, int col) {
    A->width = col;
    A->height = row;
    A->stride = col;//step
    A->elements = (double*)calloc(row * col, sizeof(double));
    for (int i = 0; i < row * col; i++)
       A->elements[i] = 0;
}

void Visualization::genRotateMatrix(Matrix A, char axis, double rad) {
    A.elements[3 * 4 + 3] = 1;
    //Rx
    if (axis == 'x') {
        A.elements[0] = 1;
        A.elements[1 * 4 + 1] = cos(rad);
        A.elements[1 * 4 + 2] = sin(rad);
        A.elements[2 * 4 + 1] = -sin(rad);
        A.elements[2 * 4 + 2] = cos(rad);
    }
    //Ry
    else if (axis == 'y') {
        A.elements[0] = cos(rad);
        A.elements[0 * 4 + 2] = -sin(rad);
        A.elements[1 * 4 + 1] = 1;
        A.elements[2 * 4 + 0] = sin(rad);
        A.elements[2 * 4 + 2] = cos(rad);
    }
    //Rz
    else if (axis == 'z') {
        A.elements[0] = cos(rad);
        A.elements[1] = sin(rad);
        A.elements[1 * 4 + 0] = -sin(rad);
        A.elements[1 * 4 + 1] = cos(rad);
        A.elements[2 * 4 + 2] = 1;
    }
}

void Visualization::genTranslationMatrix(Matrix A, double tx, double ty, double tz) {
    A.elements[0 * 4 + 3] = tx;
    A.elements[1 * 4 + 3] = ty;
    A.elements[2 * 4 + 3] = tz;
}

Visualization::Matrix Visualization::ptcRotate(Matrix A, Matrix B) {
    Matrix C;
    initMatrix(&C, A.height, B.width);
    matrixMul(A, B, C);
    //C[x,y,z,a]'
    return C;
}