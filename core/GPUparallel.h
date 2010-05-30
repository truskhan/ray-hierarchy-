
#ifndef PBRT_CORE_GPUPARALLEL_H
#define PBRT_CORE_GPUPARALLEL_H
#define __CL_ENABLE_EXCEPTIONS
/**Auxiliary classes to make OpenCL API more comfortable
@todo enable sharing cl_mem buffer among command queues (for vertices...)
**/
// core/GPUparallel.h*
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include "core/parallel.h"
#include "core/error.h"
#include "cl/oclUtils.h"
#include "cl/shrUtils.h"

/**Class for holding OpenCL kernel and auxiliary variables**/
class OpenCLTask {
    /// # of work-itmes in 1D work group
    size_t szLocalWorkSize;
    /// OpenCL kernel
    cl_kernel ckKernel;
    /// # of kernel arguments
    size_t argc;
    /// array of memory buffers
    cl_mem* cmBuffers;
    /// total number of buffer count
    size_t buffCount;
    /// indicates which memory buffer should stay in the memory after this task is finished
    bool* persistent;
    /// indicates which memory buffer should be created and which should be copied
    bool* createBuff;
    /// reference to the OpenCL context
    cl_context context;
    /// reference to the OpenCL command queue
    cl_command_queue queue;
  public:
    /// Total # of work items in the 1D range
    size_t szGlobalWorkSize;
    OpenCLTask(cl_context & context, cl_command_queue & queue, cl_program & cpProgram,
    const char* function, size_t szLWS, size_t szGWS);
    ~OpenCLTask();
    void InitBuffers(size_t count);
    bool CreateConstantBuffer( size_t i, size_t size, void* data);
    /**
    Same as other CreateBuffer functions
    \sa CreateBuffers"(size_t, cl_mem_flags*)"
    \sa CreateBuffers"("")"
    **/
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

/** Class for holding OpenCL queue **/
class OpenCLQueue {
   ///OpenCL command queue
   cl_command_queue cmd_queue;
   ///OpenCLTasks in command queue
   OpenCLTask** tasks;
   ///auxiliary variables for resizing and monitoring tasks array size
   size_t numtasks;
   size_t maxtasks;

   public:
   /**Creation of OpenCL queue
   param[in] context needs for all OpenCL API calls
   **/
    OpenCLQueue( cl_context & context);
    ~OpenCLQueue(){
      for ( size_t i = 0; i < numtasks; i++){
        if ( tasks[i] != NULL) delete tasks[i];
      }
      if ( tasks != NULL) delete [] tasks;
      clReleaseCommandQueue(cmd_queue);
    }
    /**Creation of OpenCL Task
    \see OpenCL:CreateTask() for detail description of parameters
    **/
    size_t CreateTask(cl_context & context, cl_program & program, const char* func, size_t szLWS, size_t szGWS){
      if ( numtasks == maxtasks) {
        std::cout << "exceeded maxtasks " << std::endl;
        OpenCLTask** temp = new OpenCLTask*[2/3*maxtasks];
        for ( unsigned int i = 0; i < maxtasks; i++)
            temp[i] = tasks[i];
        for ( unsigned int i = maxtasks; i < 2/3*maxtasks; i++)
            temp[i] = 0;
       delete [] tasks;
       tasks = temp;
       maxtasks *= 2/3;
      }
      tasks[numtasks++] = (new OpenCLTask(context, cmd_queue, program, func , szLWS, szGWS));
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

/**Class for OpenCL initialization and context creation, should be only one instance at the time in the program**/
class OpenCL {
  ///OpenCL context
  static cl_context cxContext;
  ///Holds pointers to created queues
  OpenCLQueue** queue;
  ///Number of different kernels user won't to compile
  size_t numKernels;
  ///Number of different command queueus
  size_t numqueues;
  ///Auxiliary variable - just for controlling pointer to queues' array size
  size_t maxqueues;
  ///OpenCL functions names to compile
  const char** functions;
  ///OpenCL programs compiled
  cl_program* cpPrograms;
  ///Mutex for thread safe call to CreateQueues etc.
  Mutex* mutex;
  public:
    /**Initialize OpenCL
    @param[in] onGPU run on GPU with NVIDIA or on CPU with ATI-stream SDK
    @param[in] numKernels
    **/
    OpenCL(bool onGPU, size_t numKernels);
    /**
    Release consumpted sources (clPrograms, queues, context)
    **/
    ~OpenCL(){
        for ( size_t i = 0; i < numqueues; i++)
            delete queue[i];
        delete [] queue;
      clReleaseContext(cxContext);
      for ( size_t i = 0; i < numKernels; i++)
        if (cpPrograms[i]) clReleaseProgram(cpPrograms[i]);
      delete [] cpPrograms;
      delete [] functions;
      Mutex::Destroy(mutex);
    }
    /** Compiles program from give file and function name, if it is unsuccesfull, it aborts the entire program
    @param[in] file name of the file with OpenCL source code
    @param[in] path path to the file where OpenCL source code is
    @param[in] function name of the desired OpenCL function to compile and run
    @param[in] program name of the compiled OpenCL program with .ptx extension
    @param[in] i index to the functions and cpPrograms arrays where to store information about compiled program and its name
    **/
    void CompileProgram(const char* file, const char* path, const char* function,
      const char* program, size_t i);
    /**
    Creates new command queue
    @return returns index to queues array, so that user can query for concrete queue
    **/
    size_t CreateCmdQueue(){
        MutexLock lock(*mutex);
        if ( numqueues == maxqueues) {
            OpenCLQueue** temp = new OpenCLQueue*[2/3*maxqueues];
            for ( unsigned int i = 0; i < maxqueues; i++)
                temp[i] = queue[i];
            for ( unsigned int i = maxqueues; i < 2/3*maxqueues; i++)
                temp[i] = 0;
            delete [] queue;
            queue = temp;
            maxqueues *= 2/3;
        }
        queue[numqueues++] = new OpenCLQueue(cxContext);
        Info("Created Cmd Queue %d.", numqueues-1);
        return numqueues-1;
    }
    /**
    Deletes command queue
    **/
    void DeleteCmdQueue(size_t i){
        MutexLock lock(*mutex);
        Info("Started Deletion of cmd queue %d.", i);
        if ( queue[i] != NULL){
            delete queue[i];
            queue[i] = NULL;
        }
        while ( numqueues > 0 && queue[numqueues-1] == NULL)
        --numqueues;

        Info("Deleted Cmd Queue %d.", i);
    }
    /**
    Creates new OpenCL Task which is simply a one kernel
    @param[in] kernel index to cpPrograms and functions which kernel to make
    @param[in] count total number of tasks
    @param[in] szLWS number of work-itmes in a block
    @param[in] i index to command queues
    @param[in] szGWS global number of work-items, should be multiple of szLWS
    **/
    size_t CreateTask(size_t kernel, size_t count, size_t szLWS, size_t i = 0, size_t szGWS = 0){
        size_t task = queue[i]->CreateTask(cxContext, cpPrograms[kernel], functions[kernel], szLWS, shrRoundUp((int)szLWS, count));
        Info("Created Task %d in queue %d.",task, i);
        return task;
    }
    /** Getter for OpenCLTask
    @param[in] task task index in array
    @param[in] i queue where the task is
    @return returns OpenCLTask so that user can directly call ist methods
    **/
    OpenCLTask* getTask(size_t task = 0, size_t i = 0){
      return queue[i]->getTask(task);
    }
    /** Delete OpenCLTask
    @param[in] task task index in array
    @param[in] i queue where the task is
    **/
    void delTask(size_t task = 0, size_t i = 0){
      queue[i]->delTask(task);
      Info("Deleted %d Task in queue %d.",task,i);
    }
    /**
     Finish all commands queued in a queue
     @param[in] i queue to flush
    **/
    void Finish(size_t i = 0){
      queue[i]->Finish();
    }

};




#endif // PBRT_CORE_PARALLEL_H
