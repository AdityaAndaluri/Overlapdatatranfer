//This is a sample implementation of updating total marks of selected students program using runbatchespipeline function which uses parallel pipelining
#include <iostream>
#include <vector>
#include<stdio.h>
#include <cuda_runtime.h>
using namespace std;

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

//structure containing all the input variables
//give all the input variables in the inputStructure

struct inputStructure {
    int numStudents;
    int* rollNumbers;
    int* mathsMarks;
    int* physicsMarks;

};

//resultant structure
//you will get results finally in this ddatastructure

struct ddatastructure{
  int* totalMarks;
  int n;
};

//Seriel function for the validation of result

void seriel(vector<int>& p , vector<int>& m, ddatastructure h){
    for(int i=0;i<h.n;i++){
      h.totalMarks[i] = p[i]+m[i];
    }
}

//function for copying of Batches for parallel implementation without pipeline
void copyBatchSync(inputStructure* s , inputStructure dbuffer , int i){
    dbuffer.numStudents = s[i].numStudents;
   checkCuda(cudaMemcpy(dbuffer.rollNumbers, s[i].rollNumbers, dbuffer.numStudents * sizeof(int), cudaMemcpyHostToDevice));
   checkCuda(cudaMemcpy(dbuffer.physicsMarks, s[i].physicsMarks, dbuffer.numStudents * sizeof(int), cudaMemcpyHostToDevice));
   checkCuda(cudaMemcpy(dbuffer.mathsMarks, s[i].mathsMarks, dbuffer.numStudents* sizeof(int), cudaMemcpyHostToDevice));
}


//function for copying of Batches for parallel implementation with pipeline

void copyBatchAsync(inputStructure* s , inputStructure dbuffer , int i , cudaStream_t* stream){
    dbuffer.numStudents = s[i].numStudents;
   checkCuda(cudaMemcpyAsync(dbuffer.rollNumbers, s[i].rollNumbers, dbuffer.numStudents * sizeof(int), cudaMemcpyHostToDevice , *stream));
   checkCuda(cudaMemcpyAsync(dbuffer.physicsMarks, s[i].physicsMarks, dbuffer.numStudents * sizeof(int), cudaMemcpyHostToDevice , *stream));
   checkCuda(cudaMemcpyAsync(dbuffer.mathsMarks, s[i].mathsMarks, dbuffer.numStudents* sizeof(int), cudaMemcpyHostToDevice , *stream));
}

//kernel function for updating the datastructure

__global__ void updateDataStructure(inputStructure dbuffer ,  ddatastructure res2 ) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid>=0 && tid < dbuffer.numStudents) {

        res2.totalMarks[dbuffer.rollNumbers[tid]] = dbuffer.mathsMarks[tid] + dbuffer.physicsMarks[tid];
    }
}

// runBatchesPipeline implementation
bool runBatchesPipleline(inputStructure* hbatch , int numBatches ,  ddatastructure res3 , inputStructure dFirstBuffer , inputStructure dSecondBuffer , void (*copyBatchAsync)(inputStructure* ,inputStructure , int , cudaStream_t*) , void(*updateDataStructure)(inputStructure ,  ddatastructure) ){

  float ms; // elapsed time in milliseconds

  // create events and streams
cudaEvent_t start,stop;
checkCuda(cudaEventCreate(&start));
checkCuda(cudaEventCreate(&stop));

cudaStream_t stream[2];
checkCuda(cudaStreamCreate(&stream[0]));
checkCuda(cudaStreamCreate(&stream[1]));

cudaEvent_t event01;
cudaEvent_t event10;

checkCuda(cudaEventCreate(&event10));
checkCuda(cudaEventCreate(&event01));

int threadsPerBlock=1024;
int blocksPerGrid;

checkCuda(cudaEventRecord(start,0));

copyBatchAsync(hbatch , dFirstBuffer , 0 , &stream[0]);
checkCuda(cudaEventRecord(event10 , stream[1])); // this can be removed and try the validation

for(int i=0;i< numBatches-1;i++){
    if(i%2==0){

      checkCuda(cudaStreamWaitEvent(stream[0],event10,0));
      blocksPerGrid = (dFirstBuffer.numStudents + threadsPerBlock - 1) / threadsPerBlock;
      updateDataStructure<<<blocksPerGrid , threadsPerBlock , 0 , stream[0]>>>(dFirstBuffer, res3);
      checkCuda(cudaEventRecord(event01 , stream[0]));

      copyBatchAsync(hbatch , dSecondBuffer , i+1 , &stream[1]);

    }
    else{

      copyBatchAsync(hbatch , dFirstBuffer , i+1 , &stream[0]);

      checkCuda(cudaStreamWaitEvent(stream[1],event01,1));
      blocksPerGrid = (dSecondBuffer.numStudents + threadsPerBlock - 1) / threadsPerBlock;
      updateDataStructure<<<blocksPerGrid , threadsPerBlock , 0 , stream[1]>>>(dSecondBuffer, res3);
      checkCuda(cudaEventRecord(event10 , stream[1]));


    }
}
if(numBatches%2==0){

      checkCuda(cudaStreamWaitEvent(stream[1],event01,1));
      blocksPerGrid = (dSecondBuffer.numStudents + threadsPerBlock - 1) / threadsPerBlock;
      updateDataStructure<<<blocksPerGrid , threadsPerBlock , 0 , stream[1]>>>(dSecondBuffer, res3);


}
else{

      checkCuda(cudaStreamWaitEvent(stream[0],event10,0));
      blocksPerGrid = (dFirstBuffer.numStudents + threadsPerBlock - 1) / threadsPerBlock;
      updateDataStructure<<<blocksPerGrid , threadsPerBlock , 0 , stream[0]>>>(dFirstBuffer, res3);

}

checkCuda(cudaDeviceSynchronize());
 checkCuda(cudaEventRecord(stop, 0)) ;
 checkCuda(cudaEventSynchronize(stop));
checkCuda(cudaEventElapsedTime(&ms, start, stop));

printf("Time for runBatchesPipleline : %f\n"  , ms);
cudaStreamDestroy(stream[0]);
cudaStreamDestroy(stream[1]);
cudaEventDestroy(event01);
cudaEventDestroy(event10);


return true;
}


int main() {
    const int numStudents = 1000000  , n = 1000000;
    const int buffersize  = 100;
    cout<<"numElements: "<<numStudents<<endl;
    cout<<"Batchsize: "<<buffersize<<endl;
    // Allocate host vectors
    vector<int> hostRollNumbers(numStudents);
    vector<int> hostMathsMarks(numStudents);
    vector<int> hostPhysicsMarks(numStudents);
  vector<int> hosttotalMarks(numStudents);

    // Generate random student details
    for (int i = 0; i < numStudents; ++i) {
        hostRollNumbers[i] = i;  // Assuming roll numbers start from 1
        hostMathsMarks[i] = rand() % 101;  // Random marks between 0 and 100
        hostPhysicsMarks[i] = rand() % 101;
    }

    //seriel implementation-----------------------------------------------------------------------------------------------------------
     ddatastructure res1;
    res1.totalMarks = (int*) malloc(numStudents*sizeof(int));
    res1.n = n;
    seriel(hostPhysicsMarks , hostMathsMarks , res1);
    //--------------------------------------------------------------------------------------------------------------------------------

    // Allocate device vectors
    inputStructure dbuffer;
    dbuffer.numStudents = buffersize;
    cudaMalloc((void**)&dbuffer.rollNumbers, buffersize * sizeof(int));
    cudaMalloc((void**)&dbuffer.mathsMarks, buffersize * sizeof(int));
    cudaMalloc((void**)&dbuffer.physicsMarks, buffersize * sizeof(int));
     ddatastructure res2;
    res2.n = numStudents;
    cudaMalloc((void**)&res2.totalMarks, numStudents * sizeof(int));

    //making hbatch[]
    int numBatches = (numStudents + buffersize - 1) / buffersize;  // Calculate the number of batches
    cout<<"Number of batches: "<<numBatches<<endl;
    inputStructure hbatch[numBatches];
    for (int i = 0; i < numBatches; i++) {
        int startIndex = i * buffersize;
        int endIndex = min((i + 1) * buffersize, n);
        int batchSize = endIndex - startIndex;

        hbatch[i].numStudents = batchSize;
        hbatch[i].rollNumbers = new int[batchSize];
        hbatch[i].physicsMarks = new int[batchSize];
        hbatch[i].mathsMarks = new int[batchSize];

        for (int j = startIndex; j < endIndex; ++j) {
            int index = j - startIndex;
            hbatch[i].rollNumbers[index] = hostRollNumbers[j];
            hbatch[i].physicsMarks[index] = hostPhysicsMarks[j];
            hbatch[i].mathsMarks[index] = hostMathsMarks[j];
        }
    }

float ms1;
cudaEvent_t start , stop;
cudaEventCreate(&start);cudaEventCreate(&stop);

float mscopy = -1;float copytime;float mscopy1 = 1000000;
float mskernel = -1;float kerneltime;float mskernel1 = 1000000;
cudaEvent_t scopy , ecopy , skernel , ekernel;
cudaEventCreate(&scopy);cudaEventCreate(&ecopy);
cudaEventCreate(&skernel);cudaEventCreate(&ekernel);

cudaEventRecord(start,0);
  for(int i=0;i<numBatches;i++){
    cudaEventRecord(scopy , 0);
    copyBatchSync(hbatch , dbuffer , i);
    cudaEventRecord(ecopy,0);
    cudaEventSynchronize(ecopy);
    cudaEventElapsedTime(&copytime, scopy , ecopy);
    mscopy = max(mscopy , copytime);
    mscopy1= min(mscopy1 , copytime);
    int blockSize = 1024;
    int gridSize = (dbuffer.numStudents + blockSize - 1) / blockSize;
    cudaEventRecord(skernel , 0);
    updateDataStructure<<<gridSize, blockSize>>>(dbuffer,res2);
    cudaDeviceSynchronize();
    cudaEventRecord(ekernel,0);
    cudaEventSynchronize(ekernel);
    cudaEventElapsedTime(&kerneltime, skernel , ekernel);
    mskernel = max(mskernel , kerneltime);
    mskernel1=min(mskernel1 , kerneltime);
  }

 cudaEventRecord(stop, 0) ;
 cudaEventSynchronize(stop);
 cudaEventElapsedTime(&ms1, start, stop);
 printf("Time for normal implementation : %f\n" , ms1);
 printf("Max Time for copy Batch  : %f\n" , mscopy);
 printf("Min Time for copy Batch :%f\n" , mscopy1);
 printf("Time for kernel execution  : %f\n" , mskernel);
printf("Min Time for kernel execute :%f\n" , mskernel1);

    // Copy device vector back to host
    cudaMemcpy(hosttotalMarks.data(), res2.totalMarks, numStudents * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0;i<numStudents;i++){
      if(res1.totalMarks[i] != hosttotalMarks[i]){cout<<"validation failed"<<endl;return 0;}
    }
    cudaFree(res2.totalMarks);
    cudaFree(dbuffer.mathsMarks);
    cudaFree(dbuffer.physicsMarks);
    cudaFree(dbuffer.rollNumbers);







//implementation 3-----------------Concentrate here. ..... this is the sample usage of our overlap data transfer------------------------------------------------------------------------------------------------

     ddatastructure res3;
    res3.n = numStudents;
    cudaMalloc((void**)&res3.totalMarks, numStudents * sizeof(int));

    inputStructure dFirstbuffer;
    dFirstbuffer.numStudents = buffersize;
    cudaMalloc((void**)&dFirstbuffer.rollNumbers, buffersize * sizeof(int));
    cudaMalloc((void**)&dFirstbuffer.mathsMarks, buffersize * sizeof(int));
    cudaMalloc((void**)&dFirstbuffer.physicsMarks, buffersize * sizeof(int));

    inputStructure dSecondbuffer;
    dSecondbuffer.numStudents = buffersize;
    cudaMalloc((void**)&dSecondbuffer.rollNumbers, buffersize * sizeof(int));
    cudaMalloc((void**)&dSecondbuffer.mathsMarks, buffersize * sizeof(int));
    cudaMalloc((void**)&dSecondbuffer.physicsMarks, buffersize * sizeof(int));

    //struct status runBatchesPipleline(struct hBatch[], int numBatches, struct dDataStructure, struct dFirstBuffer, struct dSecondBuffer,copyBatchAsync , kernel)

    if(  runBatchesPipleline(hbatch , numBatches , res3 , dFirstbuffer , dSecondbuffer , copyBatchAsync , updateDataStructure)  ){

    cudaMemcpy(hosttotalMarks.data(), res3.totalMarks, numStudents * sizeof(int), cudaMemcpyDeviceToHost);


    //for checking validation of the result we have gotten
    for(int i=0;i<numStudents;i++){
      cout << res1.totalMarks[i] << " " << hosttotalMarks[i] << "\n";
      if(res1.totalMarks[i] != hosttotalMarks[i]){cout<<"validation failed"<<endl;return 0;}
    }
    cudaFree(res3.totalMarks);
    cout<<"validation success"<<endl;
    }


    cudaFree(res3.totalMarks);
    cudaFree(dFirstbuffer.rollNumbers);
    cudaFree(dFirstbuffer.mathsMarks);
    cudaFree(dFirstbuffer.physicsMarks);

    cudaFree(dSecondbuffer.rollNumbers);
    cudaFree(dSecondbuffer.mathsMarks);
    cudaFree(dSecondbuffer.physicsMarks);

    return 0;
}
