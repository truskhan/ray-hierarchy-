
// accelerators/Naive.cpp*
#include "accelerators/naive.h"
#include "probes.h"
#include "paramset.h"
#include "intersection.h"
#include "GPUparallel.h"
#include <iostream>


using namespace std;

// NaiveAccel Method Definitions
NaiveAccel::NaiveAccel(const vector<Reference<Primitive> > &p, bool rev, bool onG) {

    triangleCount = 0;
    reverse = rev;
    onGPU = onG;
    ocl = new OpenCL(onGPU);
    ocl->CreateCmdQueue();

    for (uint32_t i = 0; i < p.size(); ++i)
        p[i]->FullyRefine(primitives);

    data[0] = new cl_float[3*3*primitives.size()];
    data[4] = new cl_float[6*primitives.size()];
    vertices = (float*) data[0];
    uvs = (float*) data[4];
    for (uint32_t i = 0; i < primitives.size(); ++i) {
        const GeometricPrimitive* gp = (dynamic_cast<const GeometricPrimitive*> (primitives[i].GetPtr()));
        if ( gp == 0 ) continue;
        const Triangle* shape = dynamic_cast<const Triangle*> (gp->GetShapePtr());
        const TriangleMesh* mesh = shape->GetMeshPtr();
        const Point &p1 = mesh->p[shape->v[0]];
        const Point &p2 = mesh->p[shape->v[1]];
        const Point &p3 = mesh->p[shape->v[2]];
        ((float*)data[0])[9*triangleCount+0] = p1.x;
        ((float*)data[0])[9*triangleCount+1] = p1.y;
        ((float*)data[0])[9*triangleCount+2] = p1.z;
        ((float*)data[0])[9*triangleCount+3] = p2.x;
        ((float*)data[0])[9*triangleCount+4] = p2.y;
        ((float*)data[0])[9*triangleCount+5] = p2.z;
        ((float*)data[0])[9*triangleCount+6] = p3[0];
        ((float*)data[0])[9*triangleCount+7] = p3[1];
        ((float*)data[0])[9*triangleCount+8] = p3[2];

        if (mesh->uvs) {
            ((float*)data[4])[6*triangleCount] = mesh->uvs[2*shape->v[0]];
            ((float*)data[4])[6*triangleCount+1] = mesh->uvs[2*shape->v[0]+1];
            ((float*)data[4])[6*triangleCount+2] = mesh->uvs[2*shape->v[1]];
            ((float*)data[4])[6*triangleCount+3] = mesh->uvs[2*shape->v[1]+1];
            ((float*)data[4])[6*triangleCount+4] = mesh->uvs[2*shape->v[2]];
            ((float*)data[4])[6*triangleCount+5] = mesh->uvs[2*shape->v[2]+1];
        } else { //todo - indicate this and compute at GPU
            ((float*)data[4])[6*triangleCount] = 0.f;
            ((float*)data[4])[6*triangleCount+1] = 0.f;
            ((float*)data[4])[6*triangleCount+2] = 1.f;
            ((float*)data[4])[6*triangleCount+3] = 0.f;
            ((float*)data[4])[6*triangleCount+4] = 1.f;
            ((float*)data[4])[6*triangleCount+5] = 1.f;
        }

        ++triangleCount;
    }

    // Compute bounds of all primitives in BVH node

    for (uint32_t i = 0; i < p.size(); ++i) {

        bbox = Union(bbox, p[i]->WorldBound());
    }

}

NaiveAccel::~NaiveAccel() {
  delete ocl;
    delete [] vertices;
    delete [] uvs;
}

BBox NaiveAccel::WorldBound() const {
#if 0
    int x = 10000;
    return BBox(Point(-x, -x, -x), Point(x, x, x));
#else
    return bbox;
#endif
}

bool NaiveAccel::Intersect(const Triangle* shape, const Ray &ray, float *tHit,
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

void NaiveAccel::Intersect(const RayDifferential *r, Intersection *in,
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

void NaiveAccel::IntersectRGPU(const RayDifferential *r, Intersection *in,
                               float* rayWeight, bool* hit, int count)  {

    size_t tn = ocl->CreateTask("../cl/intersection.cl", PbrtOptions.pbrt_path, "IntersectionR", "oclIntersection.ptx", triangleCount, 64);
    OpenCLTask* gput = ocl->getTask(tn);
    size_t b = 9;
    cl_mem_flags* flags = new cl_mem_flags[b];
    flags[0] = flags[1] = flags[2] = flags[3] = flags[4] = CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[1] = sizeof(cl_float)*3*count ; //for ray directions
    size[2] = sizeof(cl_float)*3*count; //for ray origins
    size[3] = sizeof(cl_float)*2*count; //bounds
    size[4] = sizeof(cl_float)*6*triangleCount; //uvs

    flags[5] = flags[8] = CL_MEM_READ_WRITE;
    flags[6] = flags[7] = CL_MEM_WRITE_ONLY;
    size[5] = sizeof(cl_float)*count; //for Thit
    size[6] = sizeof(cl_float)*2*count; //for tu,tv
    size[7] = sizeof(cl_float)*3*2*count; //for dpdu, dpdv
    size[8] = sizeof(int)*count; //index to shape
    gput->InitBuffers(b);
    if (!gput->CreateBuffers(size, flags)) exit(EXIT_FAILURE);
    flags[8] = CL_MEM_WRITE_ONLY;
    if (!gput->SetIntArgument(count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((int&)triangleCount)) exit(EXIT_FAILURE);

    data[1] = new cl_float[count*3]; //ray directions
    data[2] = new cl_float[count*3]; //ray origins
    data[3] = new cl_float[count*2]; //ray bounds
    data[5] = new float[count];
    data[8] = new int[count]; //for indexs to shape
    for (int k = 0; k < count; ++k) {
        ((float*)data[1])[3*k] = r[k].d[0];
        ((float*)data[1])[3*k+1] = r[k].d[1];
        ((float*)data[1])[3*k+2] = r[k].d[2];

        ((float*)data[2])[3*k] = r[k].o[0];
        ((float*)data[2])[3*k+1] = r[k].o[1];
        ((float*)data[2])[3*k+2] = r[k].o[2];

        ((float*)data[3])[2*k] = r[k].mint;
        ((float*)data[3])[2*k+1] = INFINITY;

        ((float*)data[5])[k] = INFINITY;

        ((int*)data[8])[k] = 0;
    }


    if (!gput->EnqueueWriteBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    if (!gput->EnqueueWriteBuffer(size[8], 8 , data[8])) exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);
    data[6] = new float[2*count];
    data[7] = new float[2*3*count];
    if (!gput->EnqueueReadBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    if (!gput->EnqueueReadBuffer(size[8], 8, data[8] ))exit(EXIT_FAILURE);
    int index;

    Vector dpdu, dpdv;

    for ( int i = 0; i < count; i++) {
        index = (((int*)data[8])[i]);
        hit[i] = false;
        if ( !index ) continue;
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

        dpdu = Vector(((float*)data[7])[6*i],((float*)data[7])[6*i+1],((float*)data[7])[6*i+2]);
        dpdv = Vector(((float*)data[7])[6*i+3],((float*)data[7])[6*i+4],((float*)data[7])[6*i+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](((float*)data[5])[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((float*)data[6])[2*i], ((float*)data[6])[2*i+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](((float*)data[5])[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((float*)data[6])[2*i],((float*)data[6])[2*i+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * ((float*)data[5])[i]; //thit
        r[i].maxt = ((float*)data[5])[i];
        hit[i] = true;
    }

    delete [] ((int*)data[8]);
    delete [] ((float*)data[1]);
    delete [] ((float*)data[2]);
    delete [] ((float*)data[3]);
    delete [] ((float*)data[5]);
    delete [] ((float*)data[6]);
    delete [] ((float*)data[7]);
    //cout << "RGPU" << endl;
    //exit(EXIT_SUCCESS);
}

void NaiveAccel::IntersectGPU(const RayDifferential *r, Intersection *in,
                              float* rayWeight, bool* hit, int count)  {
    size_t b;
    //bool* persistent;
    //bool* createBuff;
    cl_mem_flags* flags;
    data[1] = new cl_float[count*3]; //ray directions
    data[2] = new cl_float[count*3]; //ray origins
    void* indices = data[3] = new cl_int[count]; //indices to rays
    void* bounds = new cl_float[count*2]; //ray bounds data[3]
    size[1] = sizeof(cl_float)*3*count ; //for ray directions
    size[2] = sizeof(cl_float)*3*count; //for ray origins
    size[3] = sizeof(cl_int)*count; //for ray indices
    //cout << endl;
    for (int k = 0; k < count; ++k) {
        ((float*)data[1])[3*k] = r[k].d[0];
        ((float*)data[1])[3*k+1] = r[k].d[1];
        ((float*)data[1])[3*k+2] = r[k].d[2];
       // cout << "ray" << k << r[k].d[0] << ' ' << r[k].d[1] << ' ' << r[k].d[2] << ' ';

        ((float*)data[2])[3*k] = r[k].o[0];
        ((float*)data[2])[3*k+1] = r[k].o[1];
        ((float*)data[2])[3*k+2] = r[k].o[2];
        //cout << r[k].o[0] << ' ' << r[k].o[1] << ' ' << r[k].o[2] ;

        ((float*)bounds)[2*k] = r[k].mint;
        ((float*)bounds)[2*k+1] = INFINITY;
    }
    //cout << endl << endl;
    size_t tn = ocl->CreateTask ("../cl/raySort.cl", PbrtOptions.pbrt_path, "raySort", "oclraySort.ptx",(count+2)/3,64);
    OpenCLTask* gpusort = ocl->getTask(tn);
    b = 3;
    /*persistent = new bool[b];
    createBuff = new bool[b];
    persistent[0] = persistent[1] = true;
    createBuff[0] = createBuff[1] = true;
    gpusort.persistent = persistent;
    gpusort.createBuff = createBuff;*/
    flags = new cl_mem_flags[b];
    flags[0] = flags[1] = flags[2] = CL_MEM_READ_WRITE;
    gpusort->InitBuffers(b);
    if (!gpusort->CreateBuffers(size+1, flags)) exit(EXIT_FAILURE);
    if (!gpusort->SetIntArgument(count)) exit(EXIT_FAILURE);
    if (!gpusort->SetIntArgument(0)) exit(EXIT_FAILURE);
    flags[2] = CL_MEM_WRITE_ONLY; //indices created in the kernel
    if (!gpusort->EnqueueWriteBuffer(size+1, flags, (void**)((float**)data+1)))exit(EXIT_FAILURE);
    for ( int j = 0; j < 1; j++){
        if (!gpusort->SetIntArgument(j,4)) exit(EXIT_FAILURE);
        if (!gpusort->Run())exit(EXIT_FAILURE);
        ocl->Finish();
    }
    if (!gpusort->EnqueueReadBuffer(size+1, flags, (void**)((float**)data+1)))exit(EXIT_FAILURE);

    /*for (int k = 0; k < count; ++k) {
        cout << "ray" << k << ((float*)data[1])[3*k]
        << ' ' << ((float*)data[1])[3*k+1] << ' ' << ((float*)data[1])[3*k+2] << ' ';

        cout << ((float*)data[2])[3*k] << ' ' << ((float*)data[2])[3*k+1]
         << ' ' << ((float*)data[2])[3*k+2] ;
    }*/
    cout << endl << endl << "Indices ";
    for ( int k = 0; k < count; ++k){
      cout << ((int *)(indices))[k] << ' ';
    }
    cout << endl << endl;

    tn = ocl->CreateTask ( "../cl/intersection.cl",PbrtOptions.pbrt_path, "Intersection", "oclIntersection.ptx", count, 64);
    OpenCLTask* gput = ocl->getTask(tn);
    b = 9;
    data[3] = bounds;
    /*delete [] flags;
    delete [] persistent;
    createBuff = new bool[b];
    persistent = new bool[b];
    for ( int i = 0; i < b; i++){
      persistent[i] = false;
      createBuff[i] = true;
    }
    createBuff[1] = createBuff[2] = false;*/
    flags = new cl_mem_flags[b];
    flags[0] = flags[1] = flags[2] = flags[3] = flags[4] = CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[3] = sizeof(cl_float)*2*count; //bounds
    size[4] = sizeof(cl_float)*6*triangleCount; //uvs

    flags[5] = flags[6] = flags[7] = flags[8] = CL_MEM_WRITE_ONLY;
    size[5] = sizeof(cl_float)*gput->szGlobalWorkSize; //for Thit
    size[6] = sizeof(cl_float)*2*gput->szGlobalWorkSize; //for tu,tv
    size[7] = sizeof(cl_float)*3*2*gput->szGlobalWorkSize; //for dpdu, dpdv
    size[8] = sizeof(int)*gput->szGlobalWorkSize; //index to shape

    gput->InitBuffers(b);
    if (!gput->CreateBuffers(size, flags)) exit(EXIT_FAILURE);
    //gput.cmBuffers[1] = gpusort.cmBuffers[0];
    //gput.cmBuffers[2] = gpusort.cmBuffers[1];
    if (!gput->SetIntArgument(count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((int&)triangleCount)) exit(EXIT_FAILURE);

    if (!gput->EnqueueWriteBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);
    data[5] = new float[gput->szGlobalWorkSize];
    data[6] = new float[2*gput->szGlobalWorkSize];
    data[7] = new float[2*3*gput->szGlobalWorkSize];
    data[8] = new int[gput->szGlobalWorkSize];
    if (!gput->EnqueueReadBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    int index;

    float* tempThit = new float[gput->szGlobalWorkSize];
    float* tempT = new float[2*gput->szGlobalWorkSize];
    float* tempD = new float[2*3*gput->szGlobalWorkSize];
    int* tempIndex = new int[gput->szGlobalWorkSize];
    for ( int i = 0; i < count; i++){
      int k = ((int*)indices)[i];
      tempThit[k] = ((float*)data[5])[i];
      tempT[2*k] = ((float*)data[6])[2*i];
      tempT[2*k+1] = ((float*)data[6])[2*i+1];
      tempD[6*k] = ((float*)data[7])[6*i];
      tempD[6*k+1] = ((float*)data[7])[6*i+1];
      tempD[6*k+2] = ((float*)data[7])[6*i+2];
      tempD[6*k+3] = ((float*)data[7])[6*i+3];
      tempD[6*k+4] = ((float*)data[7])[6*i+4];
      tempD[6*k+5] = ((float*)data[7])[6*i+5];
      tempIndex[k] = ((int*)data[8])[i];
    }
    delete [] ((float*)data[5]);
    delete [] ((float*)data[6]);
    delete [] ((float*)data[7]);
    delete [] ((int*)data[8]);
    data[5] = tempThit;
    data[6] = tempT;
    data[7] = tempD;
    data[8] = tempIndex;

    Vector dpdu, dpdv;

    for ( int i = 0; i < count; i++) {
        index = (((int*)data[8])[i]);
        hit[i] = false;
        if ( !index ) continue;
        const GeometricPrimitive* p = (dynamic_cast<const GeometricPrimitive*> (primitives[index].GetPtr()));
        const Triangle* shape = dynamic_cast<const Triangle*> (p->GetShapePtr());

        dpdu = Vector(((float*)data[7])[6*i],((float*)data[7])[6*i+1],((float*)data[7])[6*i+2]);
        dpdv = Vector(((float*)data[7])[6*i+3],((float*)data[7])[6*i+4],((float*)data[7])[6*i+5]);

        // Test intersection against alpha texture, if present
        if (shape->GetMeshPtr()->alphaTexture) {
            DifferentialGeometry dgLocal(r[i](((float*)data[5])[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((float*)data[6])[2*i], ((float*)data[6])[2*i+1], shape);
            if (shape->GetMeshPtr()->alphaTexture->Evaluate(dgLocal) == 0.f)
                continue;
        }
        // Fill in _DifferentialGeometry_ from triangle hit
        in[i].dg =  DifferentialGeometry(r[i](((float*)data[5])[i]), dpdu, dpdv,
                                         Normal(0,0,0), Normal(0,0,0),
                                         ((float*)data[6])[2*i],((float*)data[6])[2*i+1], shape);
        in[i].primitive = p;
        in[i].WorldToObject = *shape->WorldToObject;
        in[i].ObjectToWorld = *shape->ObjectToWorld;
        in[i].shapeId = shape->shapeId;
        in[i].primitiveId = p->primitiveId;
        in[i].rayEpsilon = 1e-3f * ((float*)data[5])[i]; //thit
        r[i].maxt = ((float*)data[5])[i];
        hit[i] = true;
    }

    delete [] ((int*)data[8]);
    delete [] ((float*)data[1]);
    delete [] ((float*)data[2]);
    delete [] ((float*)data[3]);
    delete [] ((float*)data[5]);
    delete [] ((float*)data[6]);
    delete [] ((float*)data[7]);
}

bool NaiveAccel::Intersect(const Ray &ray, Intersection *isect) const {
    if (primitives.size() == 0) return false;
    bool hit = false;
    for (uint32_t it = 0; it < primitives.size(); ++it) {
        if (primitives[it]->Intersect(ray,isect)) {
            hit = true;
        }
    }
    return hit;
}


bool NaiveAccel::IntersectP(const Ray &ray) const {
    Intersection isect;
    return Intersect(ray, &isect);
}

void NaiveAccel::IntersectNP(const Ray* r, unsigned char* occluded, const size_t count) {
    size_t tn = ocl->CreateTask ("../cl/intersectionP.cl", PbrtOptions.pbrt_path, "IntersectionP", "oclIntersectionP.ptx", count, 64);
    OpenCLTask* gput = ocl->getTask(tn);
    size_t b = 5;
    cl_mem_flags* flags = new cl_mem_flags[b];
    flags[0] = flags[1] = flags[2] = flags[3] = CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[1] = sizeof(cl_float)*3*count ; //for ray directions
    size[2] = sizeof(cl_float)*3*count; //for ray origins
    size[3] = sizeof(cl_float)*2*count; //bounds

    flags[4] = CL_MEM_WRITE_ONLY;
    size[4] = sizeof(cl_uchar)*gput->szGlobalWorkSize; //for Thit

    gput->InitBuffers(b);
    if (!gput->CreateBuffers(size, flags)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((int&)count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((int&)triangleCount)) exit(EXIT_FAILURE);

    data[1] = new cl_float[count*3]; //ray directions
    data[2] = new cl_float[count*3]; //ray origins
    data[3] = new cl_float[count*2]; //ray bounds
    for (size_t k = 0; k < count; ++k) {
        ((float*)data[1])[3*k] = r[k].d[0];
        ((float*)data[1])[3*k+1] = r[k].d[1];
        ((float*)data[1])[3*k+2] = r[k].d[2];

        ((float*)data[2])[3*k] = r[k].o[0];
        ((float*)data[2])[3*k+1] = r[k].o[1];
        ((float*)data[2])[3*k+2] = r[k].o[2];

        ((float*)data[3])[2*k] = r[k].mint;
        ((float*)data[3])[2*k+1] = INFINITY;
    }

    data[4] = occluded;
    if (!gput->EnqueueWriteBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);
    if (!gput->EnqueueReadBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);


    delete [] ((float*)data[1]);
    delete [] ((float*)data[2]);
    delete [] ((float*)data[3]);

}

void NaiveAccel::IntersectRP(const Ray* r, unsigned char* occluded, const size_t count) {
    size_t tn = ocl->CreateTask ("../cl/intersectionP.cl", PbrtOptions.pbrt_path, "IntersectionRP", "oclIntersectionRP.ptx", count, 64);
    OpenCLTask* gput = ocl->getTask(tn);
    size_t b = 5;
    cl_mem_flags* flags = new cl_mem_flags[b];
    flags[0] = flags[1] = flags[2] = flags[3] = CL_MEM_READ_ONLY;
    size[0] = sizeof(cl_float)*3*3*triangleCount; //for vertices
    size[1] = sizeof(cl_float)*3*count ; //for ray directions
    size[2] = sizeof(cl_float)*3*count; //for ray origins
    size[3] = sizeof(cl_float)*2*count; //bounds

    flags[4] = CL_MEM_READ_WRITE;
    size[4] = sizeof(cl_uchar)*count; //for Thit
    gput->InitBuffers(b);
    if (!gput->CreateBuffers(size, flags)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((int&)count)) exit(EXIT_FAILURE);
    if (!gput->SetIntArgument((int&)triangleCount)) exit(EXIT_FAILURE);

    data[1] = new cl_float[count*3]; //ray directions
    data[2] = new cl_float[count*3]; //ray origins
    data[3] = new cl_float[count*2]; //ray bounds
    data[4] = occluded;
    for (size_t k = 0; k < count; ++k) {
        ((float*)data[1])[3*k] = r[k].d[0];
        ((float*)data[1])[3*k+1] = r[k].d[1];
        ((float*)data[1])[3*k+2] = r[k].d[2];

        ((float*)data[2])[3*k] = r[k].o[0];
        ((float*)data[2])[3*k+1] = r[k].o[1];
        ((float*)data[2])[3*k+2] = r[k].o[2];

        ((float*)data[3])[2*k] = r[k].mint;
        ((float*)data[3])[2*k+1] = INFINITY;

        ((unsigned char*)data[4])[k] = '0';
    }

    if (!gput->EnqueueWriteBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);
    if (!gput->Run())exit(EXIT_FAILURE);
    if (!gput->EnqueueReadBuffer(size, flags, (void**)data))exit(EXIT_FAILURE);


    delete [] ((float*)data[1]);
    delete [] ((float*)data[2]);
    delete [] ((float*)data[3]);

}


NaiveAccel *CreateNaiveAccelerator(const vector<Reference<Primitive> > &prims,
                                   const ParamSet &ps) {
    bool reverse = ps.FindOneBool("reverse", true);
    bool onGPU = ps.FindOneBool("onGPU",true);
    return new NaiveAccel(prims,reverse,onGPU);
}


