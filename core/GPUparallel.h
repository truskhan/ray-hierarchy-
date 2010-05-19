
#ifndef PBRT_CORE_GPUPARALLEL_H
#define PBRT_CORE_GPUPARALLEL_H
#define __CL_ENABLE_EXCEPTIONS

// core/GPUparallel.h*
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "core/parallel.h"
#include "core/error.h"
#include "cl/oclUtils.h"
#include "cl/shrUtils.h"

extern Mutex* omutex;
class OpenCLTask {
    size_t szLocalWorkSize;		    // # of work items in the 1D work group
    cl_kernel ckKernel;             // OpenCL kernel
    size_t argc; //number of kernel arguments
    cl_mem* cmBuffers;
    size_t buffCount;
    bool* persistent;
    bool* createBuff;

    cl_context context;
    cl_command_queue queue;
  public:
    size_t szGlobalWorkSize;        // Total # of work items in the 1D range
    OpenCLTask(cl_context & context, cl_command_queue & queue, cl_kernel & kernel, size_t szLWS, size_t szGWS);
    ~OpenCLTask();
    void InitBuffers(size_t count);
    bool CreateConstantBuffer( size_t i, size_t size, void* data);
    bool CreateBuffer( size_t i, size_t size, cl_mem_flags flags);
    bool CreateBuffers(size_t* size, cl_mem_flags* flags);
    bool CreateBuffers();
    void CopyBuffers(size_t srcstart, size_t srcend,
    size_t dststart, OpenCLTask* oclt);
    int SetPersistentBuff( size_t i ) { return clRetainMemObject(cmBuffers[i]);}
    bool SetIntArgument(const cl_int & arg);
    bool SetIntArgument(const cl_int & arg, const int & i);
    bool SetLocalArgument(const size_t & size);
    bool EnqueueWriteBuffer(size_t* sizes,cl_mem_flags* flags,void** data);
    bool EnqueueWriteBuffer(size_t size, size_t it, void* data);
    bool EnqueueReadBuffer(size_t* sizes,cl_mem_flags* flags,void** odata);
    bool EnqueueReadBuffer(size_t size, size_t it, void* odata);
    bool Run();
};

class OpenCLQueue {
   cl_command_queue cmd_queue; //OpenCL command queue
   OpenCLTask** tasks;
   size_t numtasks;
   size_t maxtasks;

   public:
    OpenCLQueue( cl_context & context);
    ~OpenCLQueue(){
      /*for ( size_t i = 0; i < numtasks; i++){
        if ( tasks[i] != NULL) delete tasks[i];
      }
      if ( tasks != NULL) delete [] tasks;*/
      clReleaseCommandQueue(cmd_queue);
    }
    size_t CreateTask(cl_context & context, cl_kernel & kernel, size_t szLWS, size_t szGWS){
      if ( numtasks > maxtasks) {
        std::cout << "exceeded maxtasks " << std::endl;
        abort();
      }
      tasks[numtasks++] = (new OpenCLTask(context, cmd_queue, kernel, szLWS, szGWS));
      return numtasks-1;
    }
    OpenCLTask* getTask(size_t i = 0){
       return tasks[i];
    }
    void delTask(size_t i = 0){
      if ( tasks[i] != NULL){
        delete tasks[i];
        tasks[i] = NULL;
      }
      while ( numtasks > 0 && tasks[numtasks-1] == NULL)
        --numtasks;
    }
    void Finish(){
       clFinish(cmd_queue);
    }
};

class OpenCL {
  static cl_context cxContext; //OpenCL context
  OpenCLQueue** queue;
  cl_kernel* kernels;
  size_t numKernels;
  size_t numqueues;
  size_t maxqueues;
  public:
    OpenCL(bool onGPU, size_t numKernels);
    ~OpenCL(){
        for ( size_t i = 0; i < numqueues; i++)
            delete queue[i];
        delete [] queue;
      clReleaseContext(cxContext);
      for ( size_t i = 0; i < numKernels; i++)
        if(kernels[i])clReleaseKernel(kernels[i]);
      delete [] kernels;
    }
    void CompileProgram(const char* file, const char* path, const char* function,
      const char* program, size_t i);
    size_t CreateCmdQueue(){
        MutexLock lock(*omutex);
        queue[numqueues++] = new OpenCLQueue(cxContext);
        Info("Created Cmd Queue %d.", numqueues-1);
        return numqueues-1;
    }
    void DeleteCmdQueue(size_t i){
        MutexLock lock(*omutex);
        if ( queue[i] != NULL){
            delete queue[i];
            queue[i] = NULL;
        }
        while ( numqueues > 0 && queue[numqueues-1] == NULL)
        --numqueues;

        Info("Deleted Cmd Queue %d.", i);
    }
    size_t CreateTask(size_t kernel, size_t count, size_t szLWS, size_t i = 0, size_t szGWS = 0){
        MutexLock lock(*omutex);
        size_t task = queue[i]->CreateTask(cxContext, kernels[kernel], szLWS, shrRoundUp((int)szLWS, count));
        Info("Created Task %d in queue %d.",task, i);
        return task;
    }
    OpenCLTask* getTask(size_t task = 0, size_t i = 0){
      return queue[i]->getTask(task);
    }
    void delTask(size_t task = 0, size_t i = 0){
      MutexLock lock(*omutex);
      queue[i]->delTask(task);
      Info("Deleted %d Task in queue %d.",task,i);
    }
    void Finish(size_t i = 0){
      MutexLock lock(*omutex);
      queue[i]->Finish();
    }

};




#endif // PBRT_CORE_PARALLEL_H
