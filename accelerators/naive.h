
#ifndef PBRT_ACCELERATORS_NAIVE_H
#define PBRT_ACCELERATORS_NAIVE_H

// accelerators/Naive.h*
#include "pbrt.h"
#include "primitive.h"
#include "shapes/trianglemesh.h"
#include "GPUparallel.h"

// NaiveAccel Declarations
class NaiveAccel : public Aggregate {
public:
    // NaiveAccel Public Methods
    BBox WorldBound() const;
    NaiveAccel(const vector<Reference<Primitive> > &p, bool reverse, bool onG);
    bool CanIntersect() const { return true; }
    ~NaiveAccel();
    void Intersect(const RayDifferential *r, Intersection *in, float* rayWeight, bool* hit, int counter, float* coord) const;
    void Intersect(const RayDifferential *r, Intersection *in, float* rayWeight, bool* hit, int counter){
      if (reverse) IntersectRGPU(r,in,rayWeight,hit,counter);
       else IntersectGPU(r,in,rayWeight,hit,counter);
    }
    bool Intersect(const Ray &ray, Intersection *isect) const;
    bool IntersectP(const Ray &ray) const;
    void IntersectP(const Ray* ray, unsigned char* occluded, const size_t count){
      if (reverse) IntersectRP(ray,occluded,count);
       else IntersectNP(ray,occluded,count);
    }

private:
    bool Intersect(const Triangle* shape, const Ray &ray, float *tHit,
                  Vector &dpdu, Vector &dpdv, float &tu, float &tv, float uv[3][2],const Point p[3]
                  ,float* coord) const;
    void IntersectGPU(const RayDifferential *r, Intersection *in, float* rayWeight, bool* hit, int counter);
    void IntersectRGPU(const RayDifferential *r, Intersection *in, float* rayWeight, bool* hit, int counter);
    void IntersectNP(const Ray* ray, unsigned char* occluded, const size_t count);
    void IntersectRP(const Ray* ray, unsigned char* occluded, const size_t count);
    // NaiveAccel Private Methods
    vector<Reference<Primitive> > primitives;
    BBox bbox;
    size_t triangleCount;
    void* data[9];
    size_t size[9];
    float* vertices; float* uvs;
    bool reverse;
    bool onGPU;
    OpenCL* ocl;
};


NaiveAccel *CreateNaiveAccelerator(const vector<Reference<Primitive> > &prims,
        const ParamSet &ps);

#endif // PBRT_ACCELERATORS_Naive_H
