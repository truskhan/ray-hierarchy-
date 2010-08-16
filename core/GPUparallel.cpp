#include "GPUparallel.h"
using namespace std;

cl_context OpenCL::cxContext; //OpenCL context

OpenCL::OpenCL(bool onGPU, size_t numKernels){
// create the OpenCL context on a GPU device
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
      Severe("clGetPlatformIDs failed.");
    }
    if (0 < numPlatforms)
    {
        cl_platform_id* platforms = new cl_platform_id[numPlatforms];
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
          Severe("clGetPlatformIDs failed.");
        }
        for (unsigned i = 0; i < numPlatforms; ++i)
        {
            char pbuf[100];
            status = clGetPlatformInfo(platforms[i],
                                       CL_PLATFORM_VENDOR,
                                       sizeof(pbuf),
                                       pbuf,
                                       NULL);

            if (status != CL_SUCCESS) {
              Severe("clGetPlatformIDs failed.");
            }

            platform = platforms[i];
            if ( (onGPU && !strcmp(pbuf, "NVIDIA Corporation")) ||
            (!onGPU && !strcmp(pbuf, "Advanced Micro Devices, Inc.")))
            {
                  /*cl_device_id device_id;
                  cl_int err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
                  if ( err != CL_SUCCESS)
                    Severe("Error geting device ID %i", err);
                  cl_ulong size_max;
                  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(size_max), &size_max, NULL);
                  cout << "Maxim size of local memory " << size_max << endl;
                  abort();*/
                break;
            }
        }
        delete[] platforms;
    }

    /*
     * If we could find our platform, use it. Otherwise pass a NULL and get whatever the
     * implementation thinks we should be using.
     */

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    /* Use NULL for backward compatibility */
    cl_context_properties* cprops = (NULL == platform) ? NULL : cps;

  if (onGPU)
    cxContext = clCreateContextFromType(cprops, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);
  else
    cxContext = clCreateContextFromType(cprops, CL_DEVICE_TYPE_CPU, NULL, NULL, NULL); //works only with ATI-stream SDK
  if ( cxContext == (cl_context)0){
    Severe("Error creating context ");
  }
  mutex = Mutex::Create();
  maxqueues = 100;
  numqueues = 0;
  queue = new OpenCLQueue*[maxqueues];
  for (size_t i = 0; i < maxqueues; i++)
    queue[i] = 0;

  this->numKernels = numKernels;

  cpPrograms = new cl_program[numKernels];
  functions = new const char*[numKernels];
}

OpenCLQueue::OpenCLQueue(cl_context & context){
  size_t cb;
  cl_device_id* devices;
  // get the list of GPU devices associated with context
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
  devices = new cl_device_id[cb];
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL);
  // create a command-queue
  #ifdef GPU_PROFILE
  cmd_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE , NULL); //take first device
  #else
  cmd_queue = clCreateCommandQueue(context, devices[0], NULL , NULL);
  #endif
  if (cmd_queue == (cl_command_queue)0)
  {
      clReleaseContext(context);
      free(devices);
      abort();
  }
  #ifdef GPU_PROFILE
      cl_int ciErrNum = clSetCommandQueueProperty(cmd_queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, NULL);
      if (ciErrNum != CL_SUCCESS)
          Severe(" Error %i in clSetCommandQueueProperty call !!!\n\n", ciErrNum);
  #endif
  delete [] devices;
  maxtasks = 3;
  numtasks = 0;
  tasks = new OpenCLTask*[maxtasks];
  for ( unsigned int i = 0; i < maxtasks; i++)
    tasks[i] = 0;
  globalmutex = Mutex::Create();
}

void OpenCL::CompileProgram(const char* file, const char* path, const char* function,
      const char* program, size_t i){
    char* cPathAndName;      // var for full paths to data, src, etc.
    char* cSourceCL ;         // Buffer to hold source for compilation
    size_t szKernelLength;			// Byte size of kernel code
    cl_int ciErrNum;
  #ifdef DEBUG_OUTPUT
  cout << "oclLoadProgSource (%s)...\n";
  #endif
  cPathAndName = shrFindFilePath(file, path);
  if ( cPathAndName == NULL){
    Severe( "File \"%s\" not found ",file);
  }
  cSourceCL = oclLoadProgSource(cPathAndName, "", &szKernelLength);
  if ( cSourceCL == NULL){
    Severe( "File \"%s\" not found ",file);
  }
  // Create the program
  #ifdef DEBUG_OUTPUT
  cout << "clCreateProgramWithSource...\n";
  #endif
  cpPrograms[i] = clCreateProgramWithSource(cxContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);

  ciErrNum = clBuildProgram(cpPrograms[i], 0, NULL,
  #ifdef STAT_TRIANGLE_CONE
  "-DSTAT_TRIANGLE_CONE",
  #else
      #ifdef STAT_RAY_TRIANGLE
      "-DSTAT_RAY_TRIANGLE",
      #else
      NULL,
      #endif
  #endif
  NULL, NULL);//"-g", NULL, NULL);
  if (ciErrNum != CL_SUCCESS){
    // write out standard error, Build Log and PTX, then cleanup and exit
    shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
    oclLogBuildInfo(cpPrograms[i], oclGetFirstDev(cxContext));
    oclLogPtx(cpPrograms[i], oclGetFirstDev(cxContext), program);
    Severe( "Failed building program \"%s\" !", function);
  }

  // Create the kernel
  #ifdef DEBUG_OUTPUT
  cout << "clCreateKernel ...\n";
  #endif
  functions[i] =  function;
  /*kernels[i] = clCreateKernel(cpProgram, function, &ciErrNum);
  if (ciErrNum != CL_SUCCESS){
    Severe("Invalid kerel, errNum: %d ", ciErrNum);
  }*/

}

OpenCLTask::OpenCLTask(cl_context & context, cl_command_queue & queue, Mutex* gm, cl_program & cpProgram,
 const char* function, size_t szLWS, size_t szGWS){
  this->context = context;
  this->queue = queue;
  szLocalWorkSize = szLWS;
  szGlobalWorkSize = szGWS;
  persistent = createBuff = NULL;

  cl_int ciErrNum;
  ckKernel =  clCreateKernel(cpProgram, function, &ciErrNum);
  if (ciErrNum != CL_SUCCESS){
    Severe("Invalid kerel, errNum: %d ", ciErrNum);
  }

  #define EVENTS 20
  writeEvents = new cl_event[EVENTS];
  readEvents = new cl_event[EVENTS/2];
  writeENum = readENum = 0;

  globalmutex = gm;
}

void OpenCLTask::InitBuffers(size_t count){
  argc = 0;
  cmBuffers = new cl_mem[count];
  buffCount = count;
  persistent = new bool[count];
  createBuff = new bool[count];
  for (size_t i = 0; i < count; i++) {
    persistent[i] = false;
    createBuff[i] = true;
  }
}

void OpenCLTask::CopyBuffers(size_t srcstart, size_t srcend, size_t dststart, OpenCLTask* oclt){
  size_t j = dststart;

  for ( size_t i = srcstart; i < srcend; i++){
    cmBuffers[j] = oclt->cmBuffers[i];
    createBuff[j] = false;
    j++;
  }
}

bool OpenCLTask::CreateBuffers( size_t* size, cl_mem_flags* flags){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  for ( size_t it = 0; it < buffCount; it++){
    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    if ( !createBuff[it]) continue;
    #ifdef DEBUG_OUTPUT
    cout<<"clCreateBuffer " << it << endl;
    #endif
    cmBuffers[it] = clCreateBuffer(context, flags[it], size[it], NULL, &ciErrNum);
    if ( ciErrNum != CL_SUCCESS){
      for ( size_t j = 0; j < buffCount; j++)
        if ( cmBuffers[j]) clReleaseMemObject(cmBuffers[j]);
      delete [] cmBuffers;
      cmBuffers = NULL;
      //cleanup();
      Severe("clCreateBuffer failed at buffer number %d with error %d", it, ciErrNum);
    }
  }
  ciErrNum = 0;
  for ( argc = 0; argc < buffCount; argc++){
    #ifdef DEBUG_OUTPUT
    cout << "setting argument " << argc << endl;
    #endif
    ciErrNum |= clSetKernelArg(ckKernel, argc, sizeof(cl_mem), (void*)&cmBuffers[argc]);
    if (ciErrNum != CL_SUCCESS){
      cout << "Failed setting " << argc << ". parameter " << ciErrNum <<  endl;
      return false;
    }
  }
  return true;
}

bool OpenCLTask::CreateBuffer( size_t i, size_t size, cl_mem_flags flags){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
  if ( !createBuff[i]) return true;
  #ifdef DEBUG_OUTPUT
  cout<<"clCreateBuffer " << i << endl;
  #endif
  cmBuffers[i] = clCreateBuffer(context, flags, size, NULL, &ciErrNum);
  if ( ciErrNum != CL_SUCCESS){
    for ( size_t j = 0; j < buffCount; j++)
      if ( cmBuffers[j]) clReleaseMemObject(cmBuffers[j]);
    delete [] cmBuffers;
    cmBuffers = NULL;
    Severe("clCreateBuffer failed at buffer number %d with error %d", i, ciErrNum);
  }

  ciErrNum = 0;
  #ifdef DEBUG_OUTPUT
  cout << "setting argument " << i << endl;
  #endif
  ciErrNum = clSetKernelArg(ckKernel, i, sizeof(cl_mem), (void*)&cmBuffers[argc]);
  ++argc;
  if (ciErrNum != CL_SUCCESS){
     cout << "Failed setting " << argc << ". parameter " << ciErrNum <<  endl;
     return false;
  }

  return true;
}

bool OpenCLTask::CreateConstantBuffer( size_t i, size_t size, void* data){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  #ifdef DEBUG_OUTPUT
  cout<<"clCreateConstantBuffer " << i << endl;
  #endif
  cmBuffers[i] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data, &ciErrNum);
  if ( ciErrNum != CL_SUCCESS){
      Severe("clCreateBuffer failed at constant buffer");
  }
  ciErrNum = 0;
  #ifdef DEBUG_OUTPUT
  cout << "setting argument " << i << endl;
  #endif
  ciErrNum = clSetKernelArg(ckKernel, i , sizeof(cl_mem), (void*)&cmBuffers[argc]);
  if (ciErrNum != CL_SUCCESS){
    cout << "Failed setting " << i << ". parameter " << ciErrNum <<  endl;
    return false;
  }
  argc++;

  return true;
}

bool OpenCLTask::CreateBuffers( ){
  //all buffers copied from previous task, so just set them as an argument
  cl_int ciErrNum;
  ciErrNum = 0;
  MutexLock lock(*globalmutex);
  for ( argc = 0; argc < buffCount; ++argc){
    #ifdef DEBUG_OUTPUT
    cout << "setting argument " << argc << endl;
    #endif
    ciErrNum |= clSetKernelArg(ckKernel, argc, sizeof(cl_mem), (void*)&cmBuffers[argc]);
    if (ciErrNum != CL_SUCCESS){
      cout << "Failed setting " << argc << ". parameter " << ciErrNum <<  endl;
      return false;
    }
  }
  return true;
}

bool OpenCLTask::SetIntArgument(const cl_int & arg){
  #ifdef DEBUG_OUTPUT
  cout << "set int argument " << argc << endl;
  #endif

  cl_int ciErrNum;
  ciErrNum = clSetKernelArg(ckKernel, argc++, sizeof(cl_int), (void*)&arg);
  if (ciErrNum != CL_SUCCESS){
    cout << "Failed setting parameters " << ciErrNum <<  endl;
    //cleanup();
    return false;
  }
  return true;
}

bool OpenCLTask::SetIntArgument(const cl_int & arg, const int & i){
  #ifdef DEBUG_OUTPUT
  cout << "set int argument " << endl;
  #endif
  cl_int ciErrNum;

  ciErrNum = clSetKernelArg(ckKernel, i, sizeof(cl_int), (void*)&arg);
  if (ciErrNum != CL_SUCCESS){
    cout << "Failed setting parameters " << ciErrNum <<  endl;
    //cleanup();
    return false;
  }
  return true;
}

bool OpenCLTask::SetLocalArgument(const size_t & size){
  #ifdef DEBUG_OUTPUT
  cout << "set local argument " << endl;
  #endif
  cl_int ciErrNum;

  ciErrNum = clSetKernelArg(ckKernel, argc++, size, 0);
  if (ciErrNum != CL_SUCCESS){
    cout << "Failed setting local parameters " << ciErrNum <<  endl;
    //cleanup();
    return false;
  }
  return true;
}

bool OpenCLTask::EnqueueWriteBuffer(size_t* sizes,cl_mem_flags* flags,void** data){
  size_t it = 0;
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  while ( it < buffCount ){
    if ( !createBuff[it] || flags[it] == CL_MEM_WRITE_ONLY) {
      ++it;
      continue;
    }
    ciErrNum = clEnqueueWriteBuffer(queue, cmBuffers[it], CL_FALSE, 0, sizes[it] , data[it], 0, NULL, &writeEvents[writeENum++]);
    //probably useless test, if it is asynchronous?
    if ( ciErrNum != CL_SUCCESS){
      cout << "Failed " << ciErrNum << " asynchronous data transfer at buffer "<< it << endl;
      //cleanup();
      return false;
    }
    #ifdef DEBUG
    Info("Waiting on clFinish - EnqueueWriteBuffer %d", it);
    ciErrNum = clFinish(queue);
    if ( ciErrNum != CL_SUCCESS){
      cout << "failed data transfer at buffer " << it << " error: " << ciErrNum << endl;
      //cleanup();
      return false;
    }
    #endif
    ++it;
  }
  return true;
}

bool OpenCLTask::EnqueueWriteBuffer(size_t size, size_t it, void* data){
  cl_int ciErrNum;
  Info("Before lock");
  MutexLock lock(*globalmutex);
  Info("After lock");
  ciErrNum = clEnqueueWriteBuffer(queue, cmBuffers[it], CL_FALSE, 0, size , data, 0, NULL, &writeEvents[writeENum++]);
  if ( ciErrNum != CL_SUCCESS){
    cout << "Failed " << ciErrNum << " asynchronous data transfer at buffer "<< it << endl;
    //cleanup();
    return false;
  }
  #ifdef DEBUG
  Info("Waiting on clFinish - EnqueueWriteBuffer");
  ciErrNum = clFinish(queue);
  if ( ciErrNum != CL_SUCCESS){
    Severe("failed data transfer %d", ciErrNum) ;
    //cleanup();
    return false;
  }
  #endif
  return true;
}

bool OpenCLTask::EnqueueReadBuffer(size_t size, size_t it ,void* odata){
  cl_int ciErrNum;
  MutexLock lock(*globalmutex);
  ciErrNum = clEnqueueReadBuffer(queue, cmBuffers[it], CL_TRUE, 0, size , odata, 1, &kernelEvent, &writeEvents[writeENum++]);
  if ( ciErrNum != CL_SUCCESS){
    cout << "failed read " << it << " buffer " << ciErrNum << endl;
    //cleanup();
    return false;
  }

  return true;
}

double executionTime(cl_event &event)
{
    cl_ulong start, end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

    return (double)1.0e-9 * (end - start); // convert nanoseconds to seconds on return
}

bool OpenCLTask::Run(){
    #ifdef DEBUG_OUTPUT
    cout << "clEnqueueNDRangeKernel...\n";
    #endif
    Info("Waiting on %d write events",writeENum);
    MutexLock lock(*globalmutex);
    cl_int ciErrNum;

    ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, writeENum, (writeENum == 0)?0:writeEvents,  &kernelEvent);
    //OpenCL implementace vybere velikost bloku
    //ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, NULL, 0, NULL,  NULL);
    if ( ciErrNum != CL_SUCCESS){
      Severe("failed enqueue kernel %d", ciErrNum);
    }
    #ifdef GPU_PROFILE
    Info("Waiting on clFinish - clEnqueueNDRangeKernel");
    ciErrNum = clFinish(queue);
    if ( ciErrNum != CL_SUCCESS){
      Severe("failed running kernel %d",ciErrNum);
    }
    double ktime = executionTime(kernelEvent);
    Info("Kernel execution time is %f.9", ktime);
    //clReleaseEvent(kernelEvent);
    #endif
    return true;
}

bool OpenCLTask::EnqueueReadBuffer(size_t* sizes,cl_mem_flags* flags,void** data){
  #ifdef DEBUG_OUTPUT
  cout << "clEnqueueReadBuffer (Dst)...\n\n";
  #endif
  cl_int ciErrNum;
  size_t it = 0;
  MutexLock lock(*globalmutex);
  while ( it < buffCount){
    if ( flags[it] == CL_MEM_READ_ONLY){
      it++;
      continue;
    }
    #ifdef DEBUG_OUTPUT
    cout << "Reading buffer " <<endl;
    #endif
    ciErrNum = clEnqueueReadBuffer(queue, cmBuffers[it], CL_FALSE, 0, sizes[it], data[it], 1, &kernelEvent,  &readEvents[readENum++]);
    if ( ciErrNum != CL_SUCCESS){
      cout << "failed read " << it << " buffer " << ciErrNum << endl;
      //cleanup();
      return false;
    }
    it++;
  }
  //cleanup();
  return true;
}

OpenCLTask::~OpenCLTask(){
  clReleaseKernel(ckKernel);
  if (cmBuffers){
    for ( size_t i = 0; i < buffCount; i++){
       clReleaseMemObject(cmBuffers[i]);
    }
    if (cmBuffers) delete [] cmBuffers;
    if (createBuff) delete [] createBuff;
    if (persistent) delete [] persistent;
    cmBuffers = NULL;
  }
  for ( size_t i = 0; i < writeENum; i++)
    clReleaseEvent(writeEvents[i]);
  delete [] writeEvents;
  for ( size_t i = 0; i < readENum; i++)
    clReleaseEvent(readEvents[i]);
  delete [] readEvents;
  clReleaseEvent(kernelEvent);
}

