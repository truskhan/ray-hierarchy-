

// accelerators/Naive.cpp*
#include "accelerators/rayhierarchy.h"
#include "probes.h"
#include "paramset.h"
#include "intersection.h"
#include "GPUparallel.h"
#include <iostream>


using namespace std;

void testHierarchy(float* dir, float* o, const int chunk, const int count){
  int total0 = (count+chunk-1)/chunk;
  float* cones = new float[7*(total0 + (total0+1)/2)];
  Vector r,p,x,a,q,e,c,n,g,xb,ab,rb;
  float dotrx, dotcx,t,cosfi,sinfi,cosfib, sinfib;
  uint index = 0;

  for ( int j = 0; j < count; j+=chunk, index++ ){
    //start with the zero angle enclosing cone of the first ray
    x = Vector(dir[3*j*chunk],dir[3*j*chunk+1],dir[3*j*chunk+2]);
    a = Vector(o[3*j*chunk],o[3*j*chunk+1],o[3*j*chunk+2]);
    cosfi = 1;

    for ( int i = 1; i < chunk && (j*chunk + i) < count; i++){
      r = Vector(dir[3*j*chunk+3*i],dir[3*j*chunk+3*i +1],dir[3*j*chunk+3*i+2]);
      p = Vector(o[3*j*chunk+3*i],o[3*j*chunk+3*i+1],o[3*j*chunk+3*i+2]);
      dotrx = Dot(r,x);
      if ( dotrx < cosfi){
        //extend the cone
        q = (dotrx*x - r)/Vector(dotrx*x-r).Length();
        sinfi = sin(acos(cosfi));
        e = Vector(x*cosfi + q*sinfi);
        e = e/e.Length();
        float invlength = 1/(Vector(e+r).Length());
        x = (e+r);
        x *= invlength;
        cosfi = Dot(r,x);
      }
      //check if the origin of the ray is within the wolume
      r = Vector(p-a);
      if ( r.LengthSquared() > 0.001){
        c = Vector(p-a)/Vector(p-a).Length();
        dotcx = Dot(c,x);
        if ( dotcx < cosfi){
          q = (dotcx*x - c)/Vector(dotcx*x-x).Length();
          sinfi = sin(acos(cosfi));
          e = x*cosfi + q*sinfi;
          n = x*cosfi - q*sinfi;
          g = c - Dot(n,c)*n;
          t = (g.Length()*g.Length())/Dot(e,g);
          a = a - t*e;
        }
      }
    }
    //sotre the result
    cones[7*index] = a[0];
    cones[7*index+1] = a[1];
    cones[7*index+2] = a[2];
    cones[7*index+3] = x[0];
    cones[7*index+4] = x[1];
    cones[7*index+5] = x[2];
    cones[7*index+6] = cosfi;
  }

  for ( int j = 0; j < (total0+1)/2; j+=1){
    x = Vector(cones[14*j+3],cones[14*j+4],cones[14*j+5]);
    a = Vector(cones[14*j],cones[14*j+1],cones[14*j+2]);
    cosfi = cones[14*j+6];

    ab = Vector(cones[14*j+7],cones[14*j+8],cones[14*j+9]);
    xb = Vector(cones[14*j+10],cones[14*j+11],cones[14*j+12]);
    cosfib = cones[14*j+13];
    sinfib = sin(acos(cosfib));

    cones[7*total0 + 7*j] = a.x;
    cones[7*total0 + 7*j + 1] = a.y;
    cones[7*total0 + 7*j + 2] = a.z;
    cones[7*total0 + 7*j + 3] = x.x;
    cones[7*total0 + 7*j + 4] = x.y;
    cones[7*total0 + 7*j + 5] = x.z;
    cones[7*total0 + 7*j + 6] = cosfi;
    }


}

bool intersectionAllLeaves (int j, const float* dir, const float* o, const float* bounds,
float* tHit, Vector v1, Vector v2, Vector v3, Vector e1, Vector e2, int chunk ){
    Vector s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone
    for ( int i = 0; i < chunk; i++){
      rayd = Vector(dir[3*j*chunk + 3*i], dir[3*j*chunk + 3*i+1], dir[3*j*chunk + 3*i+2]);
      rayo = Vector(o[3*j*chunk + 3*i], o[3*j*chunk +3*i+1], o[3*j*chunk + 3*i+2]);
      s1 = Cross(rayd, e2);
      divisor = Dot(s1, e1);
      if ( divisor == 0.0) continue;
      invDivisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      d = rayo - v1;
      b1 = Dot(d, s1) * invDivisor;
      if ( b1 < -1e-3f  || b1 > 1.+1e-3f) continue;

      // compute second barycentric coordinate
      s2 = Cross(d, e1);
      b2 = Dot(rayd, s2) * invDivisor;
      if ( b2 < -1e-3f || (b1 + b2) > 1.+1e-3f) continue;

      // Compute _t_ to intersection point
      t = Dot(e2, s2) * invDivisor;
      if (t < bounds[2*j*chunk + i*2]) continue;
      //mem_fence(CLK_GLOBAL_MEM_FENCE);
      if (tHit[j*chunk+i] != INFINITY && tHit[j*chunk + i] != NAN && t > tHit[j*chunk + i]) continue;
      tHit[j*chunk + i] = t;
      return true;
    }
    return false;
}

void testHierarchy1(float* dir, float* o, float* bounds, float* cones, float* vertex, float* tHit,
size_t size, size_t count, size_t chunk){
  float minx, maxx, miny, maxy, minz, maxz;
  float radius, fi, len;
  bool intersected;
  Vector center, v1, v2, v3, a , x, e1, e2 ;
  size_t total  = (count+chunk-1)/chunk;
  for ( unsigned int i = 0; i < size; i++){ //process triangles
    intersected = false;
    v1 = Vector(vertex[9*i], vertex[9*i+1], vertex[9*i+2]);
    v2 = Vector(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5]);
    v3 = Vector(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8]);
    e1 = v2 - v1;
    e2 = v3 - v1;

    minx = fmin(v1.x,v2.x);
    minx = fmin(minx,v3.x);
    maxx = fmax(v1.x,v2.x);
    maxx = fmax(maxx,v3.x);

    miny = fmin(v1.y,v2.y);
    miny = fmin(miny,v3.y);
    maxy = fmax(v1.y,v2.y);
    maxy = fmax(maxy,v3.y);

    minz = fmin(v1.z,v2.z);
    minz = fmin(minz,v3.z);
    maxz = fmax(v1.z,v2.z);
    maxz = fmax(maxz,v3.z);

    center = Vector((maxx-minx)/2,(maxy-miny)/2,(maxz-minz)/2);
    radius = fmax((maxx-minx)/2,(maxy-miny)/2);
    radius = fmax(radius,(maxz-minz)/2);

    for ( unsigned int j = 0; j < total; j++){
        intersected = false;
        a = Vector(cones[7*j],cones[7*j+1],cones[7*j+2]);
        x = Vector(cones[7*j+3],cones[7*j+4],cones[7*j+5]);
        fi = cones[7*j+6];
        len = Vector(center-a).Length();
        if (len < radius || acos(Dot(((center-a)/len),x)) - asin(radius/len)  < fi)
          intersected = intersectionAllLeaves(j, dir, o, bounds, tHit, v1,v2,v3,e1,e2,chunk);
        else {
          intersected = intersectionAllLeaves(j, dir, o, bounds, tHit, v1,v2,v3,e1,e2,chunk);
          if ( intersected) {
            cout << "skadal, co ta hierarchie? " << i << ' ' << j << endl;
          }
        }
    }

  }

}

// RayHieararchy Method Definitions
RayHieararchy::RayHieararchy(const vector<Reference<Primitive> > &p, bool onG, int chunk, int height) {

    this->chunk = chunk;
    this->height = height;
    triangleCount = 0;
    onGPU = onG;
    ocl = new OpenCL(onGPU);
    ocl->CreateCmdQueue();

    for (uint32_t i = 0; i < p.size(); ++i)
        p[i]->FullyRefine(primitives);

    data[0] = new cl_float[3*3*primitives.size()];
    data[5] = new cl_float[6*primitives.size()];
    vertices = (cl_float*) data[0];
    uvs = (cl_float*) data[5];
    for (uint32_t i = 0; i < primitives.size(); ++i) {
        const GeometricPrimitive* gp = (dynamic_cast<const GeometricPrimitive*> (primitives[i].GetPtr()));
        if ( gp == 0 ) continue;
        const Triangle* shape = dynamic_cast<const Triangle*> (gp->GetShapePtr());
        const TriangleMesh* mesh = shape->GetMeshPtr();
        const Point &p1 = mesh->p[shape->v[0]];
        const Point &p2 = mesh->p[shape->v[1]];
        const Point &p3 = mesh->p[shape->v[2]];
        ((cl_float*)data[0])[9*triangleCount+0] = p1.x;
        ((cl_float*)data[0])[9*triangleCount+1] = p1.y;
        ((cl_float*)data[0])[9*triangleCount+2] = p1.z;
        ((cl_float*)data[0])[9*triangleCount+3] = p2.x;
        ((cl_float*)data[0])[9*triangleCount+4] = p2.y;
        ((cl_float*)data[0])[9*triangleCount+5] = p2.z;
        ((cl_float*)data[0])[9*triangleCount+6] = p3[0];
        ((cl_float*)data[0])[9*triangleCount+7] = p3[1];
        ((cl_float*)data[0])[9*triangleCount+8] = p3[2];

        if (mesh->uvs) {
            ((cl_float*)data[5])[6*triangleCount] = mesh->uvs[2*shape->v[0]];
            ((cl_float*)data[5])[6*triangleCount+1] = mesh->uvs[2*shape->v[0]+1];
            ((cl_float*)data[5])[6*triangleCount+2] = mesh->uvs[2*shape->v[1]];
            ((cl_float*)data[5])[6*triangleCount+3] = mesh->uvs[2*shape->v[1]+1];
            ((cl_float*)data[5])[6*triangleCount+4] = mesh->uvs[2*shape->v[2]];
            ((cl_float*)data[5])[6*triangleCount+5] = mesh->uvs[2*shape->v[2]+1];
        } else { //todo - indicate this and compute at GPU
            ((cl_float*)data[5])[6*triangleCount] = 0.f;
            ((cl_float*)data[5])[6*triangleCount+1] = 0.f;
            ((cl_float*)data[5])[6*triangleCount+2] = 1.f;
            ((cl_float*)data[5])[6*triangleCount+3] = 0.f;
            ((cl_float*)data[5])[6*triangleCount+4] = 1.f;
            ((cl_float*)data[5])[6*triangleCount+5] = 1.f;
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

void RayHieararchy::Intersect(const RayDifferential *r, Intersection *in,
                           float* rayWeight, bool* hit, int count, float* coord) const {
    for ( int i = 0; i < count ; i++) {
        hit[i] = false;
        if ( rayWeight[i] <= 0.f) continue;
        if (primitives.size() == 0) continue;

        for (uint32_t it = 0; it < primitives.size(); ++it) {
            float thit, tu, tv;
            Vector dpdu, dpdv;
            const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[it].GetPtr()));
            if ( p == 0 ) continue;
            const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());
            float uvs[3][2];
            Point const v[3] = {shape->GetMeshPtr()->p[shape->v[0]],
                                shape->GetMeshPtr()->p[shape->v[1]],
                                shape->GetMeshPtr()->p[shape->v[2]]
                               };
            shape->GetUVs(uvs);
            if (!Intersect(shape, r[i], &thit, dpdu, dpdv, tu ,tv,uvs,v, &coord[i])) {
                continue;
            }
            *coord = it;
            // Test intersection against alpha texture, if present
            if (shape->GetMeshPtr()->alphaTexture) {
                DifferentialGeometry dgLocal(r[i](thit), dpdu, dpdv,
                                             Normal(0,0,0), Normal(0,0,0),
                                             tu, tv, shape);
                if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                    continue;
            }
            // Fill in _DifferentialGeometry_ from triangle hit
            in[i].dg =  DifferentialGeometry(r[i](thit), dpdu, dpdv,
                                             Normal(0,0,0), Normal(0,0,0),
                                             tu, tv, shape);
            in[i].primitive = p;
            in[i].WorldToObject = *shape->WorldToObject;
            in[i].ObjectToWorld = *shape->ObjectToWorld;
            in[i].shapeId = shape->shapeId;
            in[i].primitiveId = p->primitiveId;
            in[i].rayEpsilon = 1e-3f * thit;
            r[i].maxt = thit;
            hit[i] = true;
        }
    }

}

size_t RayHieararchy::ConstructRayHierarchy( cl_uint count, cl_uint chunk, cl_uint * height){
  assert(*height > 0);
  size_t b = 3;
  size_t tn = ocl->CreateTask("../cl/rayhierarchy.cl", "/home/hanci/Ploha/PBRT0704/bin/pbrt", "rayhconstruct", "oclRayhconstruct.ptx", (count+chunk-1)/chunk, 64);//zaokrouhleni nahoru
  OpenCLTask* gpuray = ocl->getTask(tn);

  flags[0] = flags[1] = CL_MEM_READ_ONLY;
  flags[2] = CL_MEM_READ_WRITE;
  size[1] = sizeof(cl_float)*3*count ; //for ray directions
  size[2] = sizeof(cl_float)*3*count; //for ray origins
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

  size[3] = sizeof(cl_float)*7*(total); //for cones - zaokrouhleno nahoru
  gpuray->InitBuffers(b);
  if (!gpuray->CreateBuffers(size+1, flags)) exit(EXIT_FAILURE);
  if (!gpuray->SetIntArgument((cl_uint)chunk)) exit(EXIT_FAILURE);
  if (!gpuray->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);

  //testHierarchy((float*)data[1], (float*)data[2],2,count);
  flags[2] = CL_MEM_WRITE_ONLY;
  if (!gpuray->EnqueueWriteBuffer(size+1, flags, (void**)((cl_float**)data+1)))exit(EXIT_FAILURE);
  if (!gpuray->Run())exit(EXIT_FAILURE);
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
  ocl->Finish();

  b = 1;
  levelcount =  (count+chunk-1)/chunk;
  size_t tasknum = ocl->CreateTask("../cl/rayhierarchy.cl", "/home/hanci/Ploha/PBRT0704/bin/pbrt", "rayLevelConstruct", "rayLevelConstruct.ptx", (levelcount+1)/2, 64);
  OpenCLTask* gpurayl = ocl->getTask(tasknum);
  gpurayl->InitBuffers(b);
  gpurayl->CopyBuffers(2,3,0,gpuray);
  if (!gpurayl->CreateBuffers( )) exit(EXIT_FAILURE);
  if (!gpurayl->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);
  if (!gpurayl->SetIntArgument((cl_uint)chunk)) exit(EXIT_FAILURE);

  for ( cl_uint i = 1; i < *height; i++){
    if (!gpurayl->SetIntArgument(i,3)) exit(EXIT_FAILURE);
    if (!gpurayl->Run())exit(EXIT_FAILURE);
    ocl->Finish();
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
  ocl->delTask(tasknum);

  return tn; //index to first task
}

//intersect computed on gpu with more rays
void RayHieararchy::Intersect(const RayDifferential *r, Intersection *in,
                               float* rayWeight, bool* hit, int count)  {

    data[1] = new cl_float[count*3]; //ray directions
    data[2] = new cl_float[count*3]; //ray origins
    data[4] = new cl_float[count*2]; //ray bounds
    data[5] = new cl_float[count];
    data[9] = new cl_uint[count]; //for index
   for (int k = 0; k < count; ++k) {
      ((cl_float*)data[1])[3*k] = r[k].d[0];
      ((cl_float*)data[1])[3*k+1] = r[k].d[1];
      ((cl_float*)data[1])[3*k+2] = r[k].d[2];

      ((cl_float*)data[2])[3*k] = r[k].o[0];
      ((cl_float*)data[2])[3*k+1] = r[k].o[1];
      ((cl_float*)data[2])[3*k+2] = r[k].o[2];

      ((cl_float*)data[4])[2*k] = r[k].mint;
      ((cl_float*)data[4])[2*k+1] = INFINITY;

      ((cl_float*)data[5])[k] = INFINITY-1; //should initialize on scene size

      ((cl_uint*)data[9])[k] = 0;
    }

    size_t tn1 = ConstructRayHierarchy(count,chunk,&height);
    OpenCLTask* gpuray = ocl->getTask(tn1);

    size_t b = 8;
    size_t tn2 = ocl->CreateTask("../cl/rayhierarchy.cl", "/home/hanci/Ploha/PBRT0704/bin/pbrt", "IntersectionR", "oclIntersection.ptx", triangleCount, 32);
    OpenCLTask* gput = ocl->getTask(tn2);
    gput->InitBuffers(b);
    gput->CopyBuffers(0,3,1,gpuray);
    ocl->delTask(tn1);


    flags[0] = flags[4] =  CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[4] = sizeof(cl_float)*2*count; //bounds

    flags[5] = flags[7] = CL_MEM_READ_WRITE;
    flags[6] = CL_MEM_WRITE_ONLY;
    size[5] = sizeof(cl_float)*count; //for Thit
    size[6] = sizeof(cl_uint)*count; //index to shape
    size[7] = sizeof(cl_int)*triangleCount*height; //stack for every thread

    if (!gput->CreateBuffers( size, flags)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)triangleCount)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)chunk)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)height)) exit(EXIT_FAILURE);

    if (!gput->EnqueueWriteBuffer(size[0], 0, data[0] ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[4], 4, data[4] ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[5], 5, data[5] ))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[6], 6, data[9] ))exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);

    data[6] = data[5];
    if (!gput->EnqueueReadBuffer(size[5], 5, data[6])) exit(EXIT_FAILURE); //Thit
    if (!gput->EnqueueReadBuffer(size[6], 6, data[9])) exit(EXIT_FAILURE); //index

    /*cout << endl << "count " << count << " Index ";
    for ( int i = 0; i < 4; i++)
      cout << (((int*)data[9])[i]) << ' ';
    abort();*/


    /*cout << endl << "Thit " ;
    for ( int i = 0; i < 21; i++){
      cout << (((float*)data[6])[i]) << ' ';
    }
    cout << endl;
    abort();*/
    /*cout << endl << "Thit " ;
    for ( int i = 0; i < triangleCount; i++){
      cout << ((float*)data[6])[i] << ' ' ;
    }
    cout << endl; abort();*/

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
    size_t tn3 = ocl->CreateTask("../cl/rayhierarchy.cl", "/home/hanci/Ploha/PBRT0704/bin/pbrt", "computeDpTuTv", "oclcomputeDpTuTv.ptx", count, 32);
    OpenCLTask* gpuRayO = ocl->getTask(tn3);
    gpuRayO->InitBuffers(b);
    gpuRayO->CopyBuffers(0,3,0,gput); // 0 vertex, 1 dir, 2 origin
    gpuRayO->CopyBuffers(6,7,3,gput); // 3 index
    ocl->delTask(tn2);

    flags[4] = CL_MEM_READ_ONLY;
    size[4] = sizeof(cl_float)*6*triangleCount; //uvs

    flags[5] = flags[6] = CL_MEM_WRITE_ONLY;
    size[5] = sizeof(cl_float)*2*count; //for tu,tv
    size[6] = sizeof(cl_float)*3*2*count; //for dpdu, dpdv

    if (!gpuRayO->CreateBuffers( size, flags)) exit(EXIT_FAILURE);
    if (!gpuRayO->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);

    if (!gpuRayO->EnqueueWriteBuffer(size[4], 4 , uvs)) exit(EXIT_FAILURE);
    if (!gpuRayO->Run())exit(EXIT_FAILURE);

    data[7] = new cl_float[2*count]; // for tu, tv
    data[8] = new cl_float[2*3*count]; // for dpdu, dpdv
    if (!gpuRayO->EnqueueReadBuffer(size[5], 5, data[7] ))exit(EXIT_FAILURE);
    if (!gpuRayO->EnqueueReadBuffer(size[6], 6, data[8] ))exit(EXIT_FAILURE);
    ocl->delTask(tn3);


    cl_uint index;
   /* cout << endl ;
    for ( int i = 0; i < count; i++) {
      index = (((int*)data[9])[i]);
      if ( index == 0) continue;
      cout << "index " << i << ". " << index << ' ' << " dpdu " << ((float*)data[8])[6*i] << ' ' << ((float*)data[8])[6*i + 1] << ' ' << ((float*)data[8])[6*i + 2] << ' '
      << ((float*)data[8])[6*i+3] << ' ' << ((float*)data[8])[6*i + 4] << ' ' << ((float*)data[8])[6*i + 5] << ' ' << endl;
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
        index = (((cl_uint*)data[9])[i]);
        hit[i] = false;
        if ( !index ) continue;
        if ( index >= triangleCount)
          continue;
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

        dpdu = Vector(((cl_float*)data[8])[6*i],((cl_float*)data[8])[6*i+1],((cl_float*)data[8])[6*i+2]);
        dpdv = Vector(((cl_float*)data[8])[6*i+3],((cl_float*)data[8])[6*i+4],((cl_float*)data[8])[6*i+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](((cl_float*)data[6])[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((cl_float*)data[7])[2*i], ((cl_float*)data[7])[2*i+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](((cl_float*)data[6])[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((cl_float*)data[7])[2*i],((cl_float*)data[7])[2*i+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * ((cl_float*)data[6])[i]; //thit
        r[i].maxt = ((cl_float*)data[6])[i];
        hit[i] = true;
    }

    delete [] ((cl_int*)data[9]);
    delete [] ((cl_float*)data[1]);
    delete [] ((cl_float*)data[2]);
    //delete [] ((float*)data[3]);
    delete [] ((cl_float*)data[4]);
    delete [] ((cl_float*)data[6]);
    delete [] ((cl_float*)data[7]);
    delete [] ((cl_float*)data[8]);
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
    Intersection isect;
    return Intersect(ray, &isect);
}

void RayHieararchy::IntersectP(const Ray* r, unsigned char* occluded, const size_t count) {

  data[1] = new cl_float[count*3]; //ray directions
  data[2] = new cl_float[count*3]; //ray origins
     // for next task store ray bounds as well
  data[4] = new cl_float[count*2]; //ray bounds
  data[5] = occluded;
  for (cl_uint k = 0; k < count; ++k) {
      ((cl_float*)data[1])[3*k] = r[k].d[0];
      ((cl_float*)data[1])[3*k+1] = r[k].d[1];
      ((cl_float*)data[1])[3*k+2] = r[k].d[2];

      ((cl_float*)data[2])[3*k] = r[k].o[0];
      ((cl_float*)data[2])[3*k+1] = r[k].o[1];
      ((cl_float*)data[2])[3*k+2] = r[k].o[2];

      ((cl_float*)data[4])[2*k] = r[k].mint;
      ((cl_float*)data[4])[2*k+1] = INFINITY;
      ((cl_uchar*)data[5])[k] = '0';
  }

    size_t tn1 = ConstructRayHierarchy(count,chunk,&height);
    OpenCLTask* gpuray = ocl->getTask(tn1);

    size_t tn2 = ocl->CreateTask ("../cl/rayhierarchy.cl", "/home/hanci/Ploha/PBRT0704/bin/pbrt", "IntersectionP", "oclIntersectionP.ptx", count, 64);
    OpenCLTask* gput = ocl->getTask(tn2);
    size_t b = 7;
    gput->InitBuffers(b);
    gput->CopyBuffers(0,3,1,gpuray);
    ocl->delTask(tn1);
    flags[0] = flags[1] = flags[2] = flags[3] = flags[4] = CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[4] = sizeof(cl_float)*2*count; //bounds

    flags[5] = flags[6] = CL_MEM_READ_WRITE;
    size[5] = sizeof(cl_uchar)*count; //for Thit
    size[6] = sizeof(cl_int)*triangleCount*height; //stack for every thread

    if (!gput->CreateBuffers(size, flags)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)triangleCount)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)chunk)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((cl_uint)height)) exit(EXIT_FAILURE);
    flags[6] = CL_MEM_WRITE_ONLY; //to prevent writing data to it
    if (!gput->EnqueueWriteBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);
    if (!gput->EnqueueReadBuffer(size[5], 5, occluded ))exit(EXIT_FAILURE);
    //if (!gput->EnqueueReadBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    //cout << "Number skipped nodes *" << (int)occluded[0] << "* " << endl; abort();

    delete [] ((cl_float*)data[1]);
    delete [] ((cl_float*)data[2]);
    //delete [] ((float*)data[3]);
    delete [] ((cl_float*)data[4]);
    ocl->delTask(tn2);
}


RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool onGPU = ps.FindOneBool("onGPU",true);
    int chunk = ps.FindOneInt("chunkSize",20);
    int height = ps.FindOneInt("height",3);
    return new RayHieararchy(prims,onGPU,chunk,height);
}


