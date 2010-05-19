
#ifndef PBRT_ACCELERATORS_RAYHIERARCHY_H
#define PBRT_ACCELERATORS_RAYHIERARCHY_H

#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "GPUparallel.h"

// RayHieararchy Declarations
class RayHieararchy : public Aggregate {
public:
    // RayHieararchy Public Methods
    BBox WorldBound() const;
    RayHieararchy(const vector<Reference<Primitive> > &p,bool onG, int chunk, int height);
    bool CanIntersect() const { return true; }
    ~RayHieararchy();
    void Intersect(const RayDifferential *r, Intersection *in, float* rayWeight, bool* hit, int counter
    #ifdef STAT_RAY_TRIANGLE
    , Spectrum *Ls
    #endif
    );
    bool Intersect(const Ray &ray, Intersection *isect) const;
    bool IntersectP(const Ray &ray) const;
    void IntersectP(const Ray* ray, unsigned char* occluded, const size_t count);
    unsigned int MaxRaysPerCall();

private:
    size_t ConstructRayHierarchy(cl_float* rayDir, cl_float* rayO, cl_uint count, cl_uint chunk, cl_uint * height, size_t cmd);
    bool Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                  Vector &dpdu, Vector &dpdv, float &tu, float &tv, float uv[3][2],const Point p[3]
                  ,float* coord) const;

    // RayHieararchy Private Methods
    vector<Reference<Primitive> > primitives;
    BBox bbox;
    size_t triangleCount;
    void* data[20];
    size_t size[20];
    cl_mem_flags flags[20];
    cl_float* vertices; cl_float* uvs;
    bool onGPU;
    cl_uint height;
    cl_uint chunk;
    OpenCL* ocl;
};


RayHieararchy *CreateRayHieararchy(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_RAYHIERARCHY_H

