
#ifndef PBRT_CORE_GPUPARALLEL_H
#define PBRT_CORE_GPUPARALLEL_H
#define __CL_ENABLE_EXCEPTIONS

// core/GPUparallel.h*
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "cl/oclUtils.h"
#include "cl/shrUtils.h"


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
    OpenCLTask(cl_context & context, cl_command_queue & queue, const char* file, const char* path, const char* function,
      const char* program, size_t szLWS, size_t szGWS);
    ~OpenCLTask();
    void InitBuffers(size_t count);
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
   //vector<OpenCLTask *> tasks;
   public:
    OpenCLQueue( cl_context & context);
    ~OpenCLQueue(){
      for ( size_t i = 0; i < numtasks; i++){
        if ( tasks[i] != NULL) delete tasks[i];
      }
      if ( tasks != NULL) delete [] tasks;
      clReleaseCommandQueue(cmd_queue);
    }
    size_t CreateTask(cl_context & context, const char* file, const char* path, const char* function, const char* program,
    size_t szLWS, size_t szGWS){
      if ( numtasks > maxtasks) {
        std::cout << "exceeded maxtasks " << std::endl;
        abort();
      }
      tasks[numtasks++] = (new OpenCLTask(context, cmd_queue, file,path,function,program, szLWS, szGWS));
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
  vector<OpenCLQueue *> queue;
  public:
    OpenCL(bool onGPU);
    ~OpenCL(){
      for ( vector<OpenCLQueue *>::iterator it = queue.begin(); it != queue.end(); it++)
        delete (*it);
      clReleaseContext(cxContext);
    }
    void CreateCmdQueue(){  queue.push_back(new OpenCLQueue(cxContext)); }
    size_t CreateTask(const char* file, const char* path, const char* function, const char* program,
      size_t count, size_t szLWS, size_t szGWS = 0, size_t i = 0){
        return queue[i]->CreateTask(cxContext, file,path,function,program, szLWS, shrRoundUp((int)szLWS, count));
    }
    OpenCLTask* getTask(size_t task = 0, size_t i = 0){
      return queue[i]->getTask(task);
    }
    void delTask(size_t task = 0, size_t i = 0){
      queue[i]->delTask(task);
    }
    void Finish(size_t i = 0){
      queue[i]->Finish();
    }

};




#endif // PBRT_CORE_PARALLEL_H
