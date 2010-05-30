

// accelerators/Naive.cpp*
#include "accelerators/rayhierarchy.h"
#include "probes.h"
#include "paramset.h"
#include "intersection.h"
#include "GPUparallel.h"
#include <iostream>
#if ( defined STAT_RAY_TRIANGLE || defined STAT_TRIANGLE_CONE)
#include "imageio.h"
#endif

#define KERNEL_COMPUTEDPTUTV 0
#define KERNEL_RAYLEVELCONSTRUCT 1
#define KERNEL_RAYCONSTRUCT 2
#define KERNEL_INTERSECTIONP 4
#define KERNEL_INTERSECTIONR 3

using namespace std;
Mutex* raymutex;
Mutex* globalmutex;
Mutex* imutex;
Mutex* ipmutex;
Mutex* dtmutex;

    /*void* data[20];
    cl_mem_flags flags[20];
*/

// RayHieararchy Method Definitions
RayHieararchy::RayHieararchy(const vector<Reference<Primitive> > &p, bool onG, int chunk, int height) {
    raymutex = Mutex::Create();
    globalmutex = Mutex::Create();
    imutex = Mutex::Create();
    ipmutex = Mutex::Create();
    dtmutex = Mutex::Create();
    this->chunk = chunk;
    this->height = height;
    triangleCount = 0;
    onGPU = onG;
    ocl = new OpenCL(onGPU,5);
    Info("Created OpenCL context");
    ocl->CompileProgram("../cl/rayhierarchy.cl", PbrtOptions.pbrt_path, "rayhconstruct", "oclRayhconstruct.ptx",KERNEL_RAYCONSTRUCT);
    ocl->CompileProgram("../cl/rayhierarchy.cl", PbrtOptions.pbrt_path, "rayLevelConstruct", "rayLevelConstruct.ptx",KERNEL_RAYLEVELCONSTRUCT);
    ocl->CompileProgram("../cl/rayhierarchy.cl", PbrtOptions.pbrt_path, "IntersectionR", "oclIntersection.ptx", KERNEL_INTERSECTIONR);
    ocl->CompileProgram("../cl/rayhierarchy.cl", PbrtOptions.pbrt_path, "computeDpTuTv", "oclcomputeDpTuTv.ptx", KERNEL_COMPUTEDPTUTV);
    ocl->CompileProgram("../cl/rayhierarchy.cl", PbrtOptions.pbrt_path, "IntersectionP", "oclIntersectionP.ptx", KERNEL_INTERSECTIONP);

    for (uint32_t i = 0; i < p.size(); ++i)
        p[i]->FullyRefine(primitives);

    vertices = new cl_float[3*3*primitives.size()];
    uvs = new cl_float[6*primitives.size()];
    for (uint32_t i = 0; i < primitives.size(); ++i) {
        const GeometricPrimitive* gp = (dynamic_cast<const GeometricPrimitive*> (primitives[i].GetPtr()));
        if ( gp == 0 ) continue;
        const Triangle* shape = dynamic_cast<const Triangle*> (gp->GetShapePtr());
        const TriangleMesh* mesh = shape->GetMeshPtr();
        const Point &p1 = mesh->p[shape->v[0]];
        const Point &p2 = mesh->p[shape->v[1]];
        const Point &p3 = mesh->p[shape->v[2]];
         vertices[9*triangleCount+0] = p1.x;
         vertices[9*triangleCount+1] = p1.y;
         vertices[9*triangleCount+2] = p1.z;
         vertices[9*triangleCount+3] = p2.x;
         vertices[9*triangleCount+4] = p2.y;
         vertices[9*triangleCount+5] = p2.z;
         vertices[9*triangleCount+6] = p3[0];
         vertices[9*triangleCount+7] = p3[1];
         vertices[9*triangleCount+8] = p3[2];

        if (mesh->uvs) {
            uvs[6*triangleCount] = mesh->uvs[2*shape->v[0]];
            uvs[6*triangleCount+1] = mesh->uvs[2*shape->v[0]+1];
            uvs[6*triangleCount+2] = mesh->uvs[2*shape->v[1]];
            uvs[6*triangleCount+3] = mesh->uvs[2*shape->v[1]+1];
            uvs[6*triangleCount+4] = mesh->uvs[2*shape->v[2]];
            uvs[6*triangleCount+5] = mesh->uvs[2*shape->v[2]+1];
        } else { //todo - indicate this and compute at GPU
            uvs[6*triangleCount] = 0.f;
            uvs[6*triangleCount+1] = 0.f;
            uvs[6*triangleCount+2] = 1.f;
            uvs[6*triangleCount+3] = 0.f;
            uvs[6*triangleCount+4] = 1.f;
            uvs[6*triangleCount+5] = 1.f;
        }

        ++triangleCount;
    }

    // Compute bounds of all primitives in BVH node

    for (uint32_t i = 0; i < p.size(); ++i) {

        bbox = Union(bbox, p[i]->WorldBound());
    }

}

RayHieararchy::~RayHieararchy() {
  delete ocl;
    delete [] vertices;
    delete [] uvs;
}

BBox RayHieararchy::WorldBound() const {
#if 0
    int x = 10000;
    return BBox(Point(-x, -x, -x), Point(x, x, x));
#else
    return bbox;
#endif
}

unsigned int RayHieararchy::MaxRaysPerCall(){
    //TODO: check the OpenCL device and decide, how many rays can be processed at once
    return 1000;
}

bool RayHieararchy::Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                           Vector &dpdu, Vector &dpdv, float& tu, float &tv,
                           float uvs[3][2], const Point p[3], float* coord) const {
    // Get triangle vertices in _p1_, _p2_, and _p3_
    const TriangleMesh* mesh = shape->GetMeshPtr();
    const Point &p1 = mesh->p[shape->v[0]];
    const Point &p2 = mesh->p[shape->v[1]];
    const Point &p3 = mesh->p[shape->v[2]];
    Vector e1 = p2 - p1;
    Vector e2 = p3 - p1;
    Vector s1 = Cross(ray.d, e2);
    float divisor = Dot(s1, e1);
    if (divisor == 0.)
        return false;
    float invDivisor = 1.f / divisor;

    // Compute first barycentric coordinate
    Vector d = ray.o - p1;
    float b1 = Dot(d, s1) * invDivisor;
    if (b1 < 0. || b1 > 1.)
        return false;

    // Compute second barycentric coordinate
    Vector s2 = Cross(d, e1);
    float b2 = Dot(ray.d, s2) * invDivisor;
    if (b2 < 0. || b1 + b2 > 1.)
        return false;

    // Compute _t_ to intersection point
    float t = Dot(e2, s2) * invDivisor;
    if (t < ray.mint || t > ray.maxt)
        return false;

    // Compute deltas for triangle partial derivatives
    float du1 = uvs[0][0] - uvs[2][0];
    float du2 = uvs[1][0] - uvs[2][0];
    float dv1 = uvs[0][1] - uvs[2][1];
    float dv2 = uvs[1][1] - uvs[2][1];
    Vector dp1 = p1 - p3, dp2 = p2 - p3;
    float determinant = du1 * dv2 - dv1 * du2;
    if (determinant == 0.f) {
        // Handle zero determinant for triangle partial derivative matrix
        CoordinateSystem(Normalize(Cross(e2, e1)), &dpdu, &dpdv);
    } else {
        float invdet = 1.f / determinant;
        dpdu = ( dv2 * dp1 - dv1 * dp2) * invdet;
        dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
    }

    // Interpolate $(u,v)$ triangle parametric coordinates
    float b0 = 1 - b1 - b2;
    tu = b0*uvs[0][0] + b1*uvs[1][0] + b2*uvs[2][0];
    tv = b0*uvs[0][1] + b1*uvs[1][1] + b2*uvs[2][1];

    *tHit = t;
    *coord = du1;
    PBRT_RAY_TRIANGLE_INTERSECTION_HIT(const_cast<Ray *>(&ray), t);
    return true;
}

size_t RayHieararchy::ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, cl_uint count, cl_uint chunk, cl_uint * height, size_t cmd){
  //MutexLock lock(*raymutex);
  assert(*height > 0);
  size_t tn = ocl->CreateTask(KERNEL_RAYCONSTRUCT, (count+chunk-1)/chunk, 64, cmd);//zaokrouhleni nahoru
  OpenCLTask* gpuray = ocl->getTask(tn,cmd);

  int total = 0;
  int levelcount = (count+chunk-1)/chunk;
  for ( cl_uint i = 0; i < *height; i++){
      total += levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
      if ( levelcount == 1 ){
        *height = i;
        break;
      }
  }
  if ( *height < 2) Severe("Too few rays for rayhierarchy! Try smaller chunks");
  gpuray->InitBuffers(3);

  Assert(gpuray->CreateBuffer(0,sizeof(cl_float)*3*count, CL_MEM_READ_ONLY )); //ray directions
  Assert(gpuray->CreateBuffer(1,sizeof(cl_float)*3*count, CL_MEM_READ_ONLY )); //ray origins
  Assert(gpuray->CreateBuffer( 2, sizeof(cl_float)*7*(total), CL_MEM_READ_WRITE)); //for cones
  Assert(gpuray->SetIntArgument((cl_uint)chunk));
  Assert(gpuray->SetIntArgument((cl_uint)count));

  Assert(gpuray->EnqueueWriteBuffer(sizeof(cl_float)*3*count, 0, rayDir));
  Assert(gpuray->EnqueueWriteBuffer(sizeof(cl_float)*3*count, 1, rayO));

  Assert(gpuray->Run());
  /*data[6] = new float[7*(total)];
  if (!gpuray->EnqueueReadBuffer(size[3], 2, data[6])) exit(EXIT_FAILURE); //cones
 // testHierarchy1(((float*)data[1]), ((float*)data[2]), ((float*)data[4]), ((float*)data[3]), ((float*)data[0]),
 // ((float*)data[5]), triangleCount, count, chunk);
  cout << endl;
  for ( int i = 0; i < (count+chunk-1)/chunk; i++){
    for ( int j = 0; j < chunk; j++){
      cout << "ray " << i*chunk+j << " dir " << ((float*)(data[1]))[3*chunk*i + 3*j + 0] << ' ' << ((float*)(data[1]))[3*chunk*i + 3*j + 1] << ' ' << ((float*)(data[1]))[3*chunk*i + 3*j + 2]
          << " o " << ((float*)(data[2]))[3*chunk*i + 3*j + 0] << ' ' << ((float*)(data[2]))[3*chunk*i + 3*j + 1] << ' ' << ((float*)(data[2]))[3*chunk*i + 3*j + 2] << endl;
    }
    cout << "cone apex " << ((float*)(data[6]))[7*i + 0]  << ' ' << ((float*)(data[6]))[7*i + 1]  << ' ' << ((float*)(data[6]))[7*i + 2]
         << " dir " << ((float*)(data[6]))[7*i + 3] << ' ' << ((float*)(data[6]))[7*i + 4]  << ' ' << ((float*)(data[6]))[7*i + 5]
         << " angle " << ((float*)(data[6]))[7*i + 6] << endl;
  }
  abort();*/

  int err = 0;
  err != gpuray->SetPersistentBuff(0);
  err != gpuray->SetPersistentBuff(1);
  err != gpuray->SetPersistentBuff(2);
  if ( err != CL_SUCCESS) {
    cout << "can't set persistent cl_mem object " << endl;
    abort();
  }
  ocl->Finish(cmd);

  size_t b = 1;
  levelcount =  (count+chunk-1)/chunk;
  size_t tasknum = ocl->CreateTask(KERNEL_RAYLEVELCONSTRUCT, (levelcount+1)/2, 64, cmd);
  OpenCLTask* gpurayl = ocl->getTask(tasknum,cmd);
  gpurayl->InitBuffers(b);
  gpurayl->CopyBuffers(2,3,0,gpuray);
  if (!gpurayl->CreateBuffers( )) exit(EXIT_FAILURE);
  if (!gpurayl->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);
  if (!gpurayl->SetIntArgument((cl_uint)chunk)) exit(EXIT_FAILURE);

  for ( cl_uint i = 1; i < *height; i++){
    if (!gpurayl->SetIntArgument(i,3)) exit(EXIT_FAILURE);
    if (!gpurayl->Run())exit(EXIT_FAILURE);
    ocl->Finish(cmd);
  }

  //data[6] = new float[7*(total)];
  //if (!gpurayl->EnqueueReadBuffer(size[3], 0, data[6])) exit(EXIT_FAILURE); //cones

  /*cout << endl;
  int t = (count+chunk-1)/chunk;
  t *= 7;
  for ( int i = 0; i < (count+chunk-1)/chunk; i++){
    for ( int j = 0; j < 2; j++){
      cout << "cone apex " << ((float*)(data[6]))[14*i + 7*j + 0]  << ' ' << ((float*)(data[6]))[14*i + 7*j + 1]  << ' ' << ((float*)(data[6]))[14*i + 7*j + 2]
           << " dir " << ((float*)(data[6]))[14*i + 7*j + 3] << ' ' << ((float*)(data[6]))[14*i + 7*j + 4]  << ' ' << ((float*)(data[6]))[14*i + 7*j + 5]
           << " angle " << ((float*)(data[6]))[14*i + 7*j + 6] << endl;
    }
    cout << "TOTAL cone apex " << ((float*)(data[6]))[t + 7*i + 0]  << ' ' << ((float*)(data[6]))[t + 7*i + 1]  << ' ' << ((float*)(data[6]))[t + 7*i + 2]
         << " dir " << ((float*)(data[6]))[t + 7*i + 3] << ' ' << ((float*)(data[6]))[t + 7*i + 4]  << ' ' << ((float*)(data[6]))[t + 7*i + 5]
         << " angle " << ((float*)(data[6]))[t + 7*i + 6] << endl;
  }
  abort();*/

  err = gpurayl->SetPersistentBuff(0);
  if ( err != CL_SUCCESS) {
    cout << "can't set persistent cl_mem object " << endl;
    abort();
  }
  ocl->delTask(tasknum, cmd);

  return tn; //index to first task
}

//intersect computed on gpu with more rays
void RayHieararchy::Intersect(const RayDifferential *r, Intersection *in,
                               float* rayWeight, bool* hit, int count
    #ifdef STAT_RAY_TRIANGLE
    , Spectrum *Ls
    #endif
    )  {
//MutexLock lock(*globalmutex);

    size_t* size = new size_t[20];
    size_t cmd = ocl->CreateCmdQueue();
    cout << "count " << count << endl;
    cl_float* rayDirArray = new cl_float[count*3]; //ray directions data[1]
    cl_float* rayOArray = new cl_float[count*3]; //ray origins data[2]
    cl_float* rayBoundsArray = new cl_float[count*2]; //ray bounds data[4]
    cl_float* tHitArray = new cl_float[count]; //data[5]
    cl_uint* indexArray = new cl_uint[count]; //for index ALLOCA(cl_uint, count); data[9]
    #ifdef STAT_RAY_TRIANGLE
    data[10] = new float[count];
    #endif

   for (int k = 0; k < count; ++k) {
      rayDirArray[3*k] = r[k].d[0];
      rayDirArray[3*k+1] = r[k].d[1];
      rayDirArray[3*k+2] = r[k].d[2];

      rayOArray[3*k] = r[k].o[0];
      rayOArray[3*k+1] = r[k].o[1];
      rayOArray[3*k+2] = r[k].o[2];

      rayBoundsArray[2*k] = r[k].mint;
      rayBoundsArray[2*k+1] = INFINITY;

      tHitArray[k] = INFINITY-1; //should initialize on scene size

      indexArray[k] = 0;
      #ifdef STAT_RAY_TRIANGLE
      ((cl_uint*)data[10])[k] = 0;
      #endif
    }

//MutexLock lock(*imutex);
    size_t tn1 = ConstructRayHierarchy(rayDirArray, rayOArray, count,chunk,&height, cmd);
    OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

    size_t b = 7;
    #ifdef STAT_TRIANGLE_CONE
     ++b;
    #endif
    #ifdef STAT_RAY_TRIANGLE
    ++b;
    #endif
    size_t tn2 = ocl->CreateTask(KERNEL_INTERSECTIONR, triangleCount, 32, cmd);
    OpenCLTask* gput = ocl->getTask(tn2,cmd);
    gput->InitBuffers(b);
    gput->CopyBuffers(0,3,1,gpuray);
    ocl->delTask(tn1,cmd);
    cl_mem_flags flags[20];


    flags[0] = flags[4] =  CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[4] = sizeof(cl_float)*2*count; //bounds

    flags[5] = flags[7] = CL_MEM_READ_WRITE;
    flags[6] = CL_MEM_WRITE_ONLY;
    size[5] = sizeof(cl_float)*count; //for Thit
    size[6] = sizeof(cl_uint)*count; //index to shape
    //size[7] = sizeof(cl_int)*triangleCount*(height+2); //stack for every thread
    b = 6;
    #ifdef STAT_TRIANGLE_CONE
    size[++b] = sizeof(cl_uint)*triangleCount;
    flags[b] = CL_MEM_WRITE_ONLY;
    #endif
    #ifdef STAT_RAY_TRIANGLE
    size[++b] = sizeof(cl_uint)*count;
    flags[b] = CL_MEM_WRITE_ONLY;
    #endif

    Assert(gput->CreateBuffers( size, flags));
    //should be workGroupSize instead of 32
    Assert(gput->SetLocalArgument(sizeof(cl_int)*32*(height))); //stack for every thread
    Assert(gput->SetIntArgument((cl_int)count));
    Assert(gput->SetIntArgument((cl_int)triangleCount));
    Assert(gput->SetIntArgument((cl_int)chunk));
    Assert(gput->SetIntArgument((cl_int)height));

    if (!gput->EnqueueWriteBuffer(size[0], 0, vertices ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[4], 4, rayBoundsArray ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[5], 5, tHitArray ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[6], 6, indexArray ))exit(EXIT_FAILURE);
    #ifdef STAT_RAY_TRIANGLE
    Assert(gput->EnqueueWriteBuffer(size[b], b, data[10]));
    #endif

    if (!gput->Run())exit(EXIT_FAILURE);

    if (!gput->EnqueueReadBuffer(size[5], 5, tHitArray)) exit(EXIT_FAILURE); //Thit data[6], data[5]
    if (!gput->EnqueueReadBuffer(size[6], 6, indexArray)) exit(EXIT_FAILURE); //index data[9]
    #ifdef STAT_RAY_TRIANGLE
    Assert(gput->EnqueueReadBuffer(size[b], b, data[10]));
    #ifdef DEBUG_OUTPUT
    cout << endl << "ray triangle intersections test ";
    #endif
    uint i = 0;
    uint temp;
    for (i = 0; i < count; i++){
        temp = Clamp(((cl_uint*)data[10])[i],0,255) ;
        Ls[i] = Spectrum(temp);
        #ifdef DEBUG_OUTPUT
        cout << temp << ' ';
        #endif
    }
    #ifdef DEBUG_OUTPUT
    cout << endl;
    #endif
    delete [] ((uint*)data[10]);
    return;
    #endif
    #ifdef STAT_TRIANGLE_CONE
    data[10] = new cl_uint[triangleCount];
    Assert(gput->EnqueueReadBuffer(size[8], 8, data[10]));
    cout << endl << "triangle cone intersections ";
    uint i = 0;
    for ( i = 0; i < triangleCount; i++)
      cout << (((uint*)data[10])[i]) << ' ';
    cout <<  endl;
    abort();
    #endif

    /*cout << endl << "count " << count << " Index ";
    for ( int i = 0; i < 4; i++)
      cout << (((int*)data[9])[i]) << ' ';
    abort();*/


   /* cout << endl << "Thit " ;
    for ( int i = 0; i < 768; i++){
      cout << "i" << i << ": " << (((float*)data[6])[i]) << ' ';
    }
    cout << endl;
    abort();*/
   /* cout << endl << "Thit " ;
    for ( int i = 0; i < count; i++){
      if ( indexArray[i] == 0) continue;
      cout << tHitArray[i] << ' ' ;
    }
    cout << endl;*/

    int err = 0;
    err != gput->SetPersistentBuff(0);//vertex
    err != gput->SetPersistentBuff(1);//dir
    err != gput->SetPersistentBuff(2);//origin
    err != gput->SetPersistentBuff(6);//index
    if ( err != CL_SUCCESS) {
      cout << "can't set persistent cl_mem object " << endl;
      abort();
    }

    b = 7;
    size_t tn3 = ocl->CreateTask(KERNEL_COMPUTEDPTUTV, count, 32, cmd);
    OpenCLTask* gpuRayO = ocl->getTask(tn3,cmd);
    gpuRayO->InitBuffers(b);
    gpuRayO->CopyBuffers(0,3,0,gput); // 0 vertex, 1 dir, 2 origin
    gpuRayO->CopyBuffers(6,7,3,gput); // 3 index
    ocl->delTask(tn2,cmd);

    flags[4] = CL_MEM_READ_ONLY;
    size[4] = sizeof(cl_float)*6*triangleCount; //uvs

    flags[5] = flags[6] = CL_MEM_WRITE_ONLY;
    size[5] = sizeof(cl_float)*2*count; //for tu,tv
    size[6] = sizeof(cl_float)*3*2*count; //for dpdu, dpdv

    if (!gpuRayO->CreateBuffers( size, flags)) exit(EXIT_FAILURE);
    if (!gpuRayO->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);

    if (!gpuRayO->EnqueueWriteBuffer(size[4], 4 , uvs)) exit(EXIT_FAILURE);
    if (!gpuRayO->Run())exit(EXIT_FAILURE);

    Info("count %d", count);
    cl_float* tutvArray = new cl_float[2*count]; // for tu, tv data[7]
    cl_float* dpduArray = new cl_float[2*3*count]; // for dpdu, dpdv data[8]
    if (!gpuRayO->EnqueueReadBuffer(size[5], 5, tutvArray ))exit(EXIT_FAILURE);
    if (!gpuRayO->EnqueueReadBuffer(size[6], 6, dpduArray ))exit(EXIT_FAILURE);
    ocl->delTask(tn3,cmd);


    cl_uint index;
    /*cout << endl ;
    for ( int j = 0; j < count; j++) {
      index = indexArray[j];
      if ( index == 0) continue;
      cout << "index " << j << ". " << index << ' ' << " dpdu " << dpduArray[6*j] << ' ' << dpduArray[6*j + 1] << ' ' << dpduArray[6*j + 2] << ' '
      << dpduArray[6*j+3] << ' ' << dpduArray[6*j + 4] << ' ' << dpduArray[6*j + 5] << ' ' << endl;
    }
    cout << endl;*/

    /*cout << endl ;
    for ( int i = 0; i < count; i++) {
      index = (((int*)data[9])[i]);
      if ( index == 0) continue;
      cout << "ray " << i << ". intersected by triangle " << index << endl;
    }
    cout << endl;
    abort();*/

    Vector dpdu, dpdv;

    for ( int i = 0; i < count; i++) {
        index = indexArray[i];
        hit[i] = false;
        if ( !index ) continue;
        Assert( index < triangleCount);
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

        dpdu = Vector(dpduArray[6*i],dpduArray[6*i+1],dpduArray[6*i+2]);
        dpdv = Vector(dpduArray[6*i+3],dpduArray[6*i+4],dpduArray[6*i+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](tHitArray[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         tutvArray[2*i], tutvArray[2*i+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](tHitArray[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         tutvArray[2*i],tutvArray[2*i+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * tHitArray[i]; //thit
        r[i].maxt = tHitArray[i];
        hit[i] = true;
        PBRT_RAY_TRIANGLE_INTERSECTION_HIT(&r[i], r[i].maxt);
    }

    ocl->delTask(tn1,cmd);
    ocl->delTask(tn2,cmd);
    ocl->delTask(tn3,cmd);
    ocl->DeleteCmdQueue(cmd);
   // delete [] ((cl_int*)data[9]);
    delete [] rayDirArray; //(cl_float*)data[1]);
    delete [] rayOArray; //((cl_float*)data[2]);
    //delete [] ((float*)data[3]);
    delete [] rayBoundsArray; //((cl_float*)data[4]);
    delete [] tHitArray; //((cl_float*)data[6]);
    delete [] tutvArray; //((cl_float*)data[7]);
    delete [] dpduArray; //((cl_float*)data[8]);
    delete [] size;
}

bool RayHieararchy::Intersect(const Ray &ray, Intersection *isect) const {
    if (primitives.size() == 0) return false;
    bool hit = false;
    for (uint32_t it = 0; it < primitives.size(); ++it) {
        if (primitives[it]->Intersect(ray,isect)) {
            hit = true;
        }
    }
    return hit;
}


bool RayHieararchy::IntersectP(const Ray &ray) const {
    /*Intersection isect;
    return Intersect(ray, &isect);*/
    if (primitives.size() == 0) return false;
    bool hit = false;
    for (uint32_t it = 0; it < primitives.size(); ++it) {
        if (primitives[it]->IntersectP(ray)) {
            hit = true;
            PBRT_RAY_TRIANGLE_INTERSECTIONP_HIT(&ray,0);
        }
        PBRT_FINISHED_RAY_INTERSECTIONP(&ray, int(hit));
    }
    return hit;
}

void RayHieararchy::IntersectP(const Ray* r, unsigned char* occluded, const size_t count) {

//MutexLock lock1(*globalmutex);
  size_t cmd = ocl->CreateCmdQueue();
  cl_float* rayDirArray = new cl_float[count*3]; //ray directions
  cl_float* rayOArray = new cl_float[count*3]; //ray origins
     // for next task store ray bounds as well
  cl_float* rayBoundsArray = new cl_float[count*2]; //ray bounds
  size_t* size = new size_t[20];
  cout << "count " << count << endl;
  for (cl_uint k = 0; k < count; ++k) {
      rayDirArray[3*k] = r[k].d[0];
      rayDirArray[3*k+1] = r[k].d[1];
      rayDirArray[3*k+2] = r[k].d[2];

      rayOArray[3*k] = r[k].o[0];
      rayOArray[3*k+1] = r[k].o[1];
      rayOArray[3*k+2] = r[k].o[2];

      rayBoundsArray[2*k] = r[k].mint;
      rayBoundsArray[2*k+1] = INFINITY;
      occluded[k] = '0';
  }
//MutexLock lock(*ipmutex);
    size_t tn1 = ConstructRayHierarchy(rayDirArray, rayOArray, count,chunk,&height, cmd);
    OpenCLTask* gpuray = ocl->getTask(tn1,cmd);

    Info("height of the ray hieararchy in IntersectP is %d",height);
    size_t tn2 = ocl->CreateTask (KERNEL_INTERSECTIONP, count, 64, cmd);
    OpenCLTask* gput = ocl->getTask(tn2,cmd);
    size_t b = 6;
    gput->InitBuffers(b);
    gput->CopyBuffers(0,3,1,gpuray);
    ocl->delTask(tn1,cmd);
    cl_mem_flags flags[20];
    flags[0] = flags[1] = flags[2] = flags[3] = flags[4] = CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[4] = sizeof(cl_float)*2*count; //bounds

    flags[5] = CL_MEM_READ_WRITE;
    size[5] = sizeof(cl_uchar)*count; //for Thit
    //size[6] = sizeof(cl_int)*triangleCount*height; //stack for every thread

    if (!gput->CreateBuffers(size, flags)) exit(EXIT_FAILURE);
    if (!gput->SetLocalArgument(sizeof(cl_int)*32*height)); //stack for every thread
    if (!gput->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)triangleCount)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)chunk)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)height)) exit(EXIT_FAILURE);

    if (!gput->EnqueueWriteBuffer(size[0], 0, vertices ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[4], 4, rayBoundsArray ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[5], 5, occluded ))exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);
    if (!gput->EnqueueReadBuffer(size[5], 5, occluded ))exit(EXIT_FAILURE);

    #ifdef PBRT_STATS_COUNTERS
    for ( unsigned i = 0; i < count; i++){
        PBRT_FINISHED_RAY_INTERSECTIONP(&r[i], int(occluded[i]));
        if (occluded[i] != '0')
            PBRT_RAY_TRIANGLE_INTERSECTIONP_HIT(&r[i],0);
    }
    #endif

    delete [] rayDirArray;
    delete [] rayOArray;
    delete [] rayBoundsArray;
    delete [] size;
    ocl->delTask(tn2,cmd);
    ocl->delTask(tn1,cmd);
    ocl->DeleteCmdQueue(cmd);
}


RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool onGPU = ps.FindOneBool("onGPU",true);
    int chunk = ps.FindOneInt("chunkSize",20);
    int height = ps.FindOneInt("height",3);
    return new RayHieararchy(prims,onGPU,chunk,height);
}


