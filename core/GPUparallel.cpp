#include "GPUparallel.h"
#include "core/error.h"
using namespace std;

cl_context OpenCL::cxContext; //OpenCL context

OpenCL::OpenCL(bool onGPU){
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
            if ( !strcmp(pbuf, "NVIDIA Corporation") ||
             !strcmp(pbuf, "Advanced Micro Devices, Inc."))
            {
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

}

OpenCLQueue::OpenCLQueue(cl_context & context){
  size_t cb;
  cl_device_id* devices;
  // get the list of GPU devices associated with context
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
  devices = new cl_device_id[cb];
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL);
  // create a command-queue
  cmd_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE , NULL); //take first device
  if (cmd_queue == (cl_command_queue)0)
  {
      clReleaseContext(context);
      free(devices);
      abort();
  }
  delete [] devices;
  maxtasks = 100;
  numtasks = 0;
  tasks = new OpenCLTask*[maxtasks];
  for ( unsigned int i = 0; i < maxtasks; i++)
    tasks[i] = 0;
}

OpenCLTask::OpenCLTask(cl_context & context, cl_command_queue & queue, const char* file,
  const char* path, const char* function, const char* program, size_t szLWS, size_t szGWS){
    this->context = context;
    this->queue = queue;
    szLocalWorkSize = szLWS;
    szGlobalWorkSize = szGWS;
    persistent = createBuff = NULL;
    cl_program cpProgram;           // OpenCL program
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
  cpProgram = clCreateProgramWithSource(context, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);

  ciErrNum = clBuildProgram(cpProgram, 0, NULL,
  #ifdef STAT_TRIANGLE_CONE
  "-DSTAT_TRIANGLE_CONE -g",
  #else
      #ifdef STAT_RAY_TRIANGLE
      " -DSTAT_RAY_TRIANGLE -g",
      #else
      NULL,
      #endif
  #endif
  NULL, NULL);//"-g", NULL, NULL);
  if (ciErrNum != CL_SUCCESS){
    // write out standard error, Build Log and PTX, then cleanup and exit
    shrLog(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
    oclLogBuildInfo(cpProgram, oclGetFirstDev(context));
    oclLogPtx(cpProgram, oclGetFirstDev(context), program);
    Severe( "Failed building program \"%s\" !", function);
  }

  // Create the kernel
  #ifdef DEBUG_OUTPUT
  cout << "clCreateKernel ...\n";
  #endif
  ckKernel = clCreateKernel(cpProgram, function, &ciErrNum);
  if (ciErrNum != CL_SUCCESS){
    Severe("Invalid kerel, errNum: %d ", ciErrNum);
  }

}

void OpenCLTask::InitBuffers(size_t count){
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
      Severe("clCreateBuffer failed at buffer number %d", it);
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

bool OpenCLTask::CreateBuffers( ){
  //all buffers copied from previous task, so just set them as an argument
  cl_int ciErrNum;
  ciErrNum = 0;
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
  while ( it < buffCount ){
    if ( !createBuff[it] || flags[it] == CL_MEM_WRITE_ONLY) {
      ++it;
      continue;
    }
    ciErrNum = clEnqueueWriteBuffer(queue, cmBuffers[it], CL_FALSE, 0, sizes[it] , data[it], 0, NULL, NULL);
    if ( ciErrNum != CL_SUCCESS){
      cout << "Failed " << ciErrNum << " asynchronous data transfer at buffer "<< it << endl;
      //cleanup();
      return false;
    }
    ++it;
  }
  return true;
}

bool OpenCLTask::EnqueueWriteBuffer(size_t size, size_t it, void* data){
  cl_int ciErrNum;
  ciErrNum = clEnqueueWriteBuffer(queue, cmBuffers[it], CL_FALSE, 0, size , data, 0, NULL, NULL);
  if ( ciErrNum != CL_SUCCESS){
    cout << "Failed " << ciErrNum << " asynchronous data transfer at buffer "<< it << endl;
    //cleanup();
    return false;
  }
  return true;
}

bool OpenCLTask::EnqueueReadBuffer(size_t size, size_t it ,void* odata){

  cl_int ciErrNum;
  ciErrNum = clEnqueueReadBuffer(queue, cmBuffers[it], CL_TRUE, 0, size , odata, 0, NULL, NULL);
  if ( ciErrNum != CL_SUCCESS){
    cout << "failed read " << it << " buffer " << ciErrNum << endl;
    //cleanup();
    return false;
  }

  return true;
}

bool OpenCLTask::Run(){
    #ifdef DEBUG_OUTPUT
    cout << "clEnqueueNDRangeKernel...\n";
    #endif
    cl_int ciErrNum;
    ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL,  NULL);
    //OpenCL implementace vybere velikost bloku
    //ciErrNum = clEnqueueNDRangeKernel(queue, ckKernel, 1, NULL, &szGlobalWorkSize, NULL, 0, NULL,  NULL);
    if ( ciErrNum != CL_SUCCESS){
      cout << "failed enqueue kernel " << ciErrNum << endl;
      //cleanup();
      return false;
    }
    return true;
}

bool OpenCLTask::EnqueueReadBuffer(size_t* sizes,cl_mem_flags* flags,void** data){
  #ifdef DEBUG_OUTPUT
  cout << "clEnqueueReadBuffer (Dst)...\n\n";
  #endif
  cl_int ciErrNum;
  size_t it = 0;
  while ( it < buffCount){
    if ( flags[it] == CL_MEM_READ_ONLY){
      it++;
      continue;
    }
    #ifdef DEBUG_OUTPUT
    cout << "Reading buffer " <<endl;
    #endif
    ciErrNum = clEnqueueReadBuffer(queue, cmBuffers[it], CL_TRUE, 0, sizes[it], data[it], 0, NULL,  NULL);
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
  if(ckKernel)clReleaseKernel(ckKernel);
  //if(ceEvent)clReleaseEvent(ceEvent);
  //if(cpProgram)clReleaseProgram(cpProgram);
  if (cmBuffers){
    for ( size_t i = 0; i < buffCount; i++){
       clReleaseMemObject(cmBuffers[i]);
    }
    if (cmBuffers) delete [] cmBuffers;
    if (persistent) delete [] persistent;
    cmBuffers = NULL;
  }
}

