

#ifndef PBRT_RENDERERS_GPURENDERER_H
#define PBRT_RENDERERS_GPURENDERER_H

// renderers/GPURenderer.h*
#include "pbrt.h"
#include "renderer.h"
#include "parallel.h"

// GPURenderer Declarations
class breadthFirst : public Renderer {
public:
    // GPURenderer Public Methods
    GPURenderer(Sampler *s, Camera *c, SurfaceIntegrator *si,
        VolumeIntegrator *vi);
    ~GPURenderer();
    void Render(const Scene *scene);
    Spectrum Li(const Scene *scene, const RayDifferential &ray,
        const Sample *sample, RNG &rng, MemoryArena &arena,
        Intersection *isect = NULL, Spectrum *T = NULL) const;
    void Li(const Scene *scene, const RayDifferential* ray,
        const Sample *sample, RNG &rng, MemoryArena &arena,
        Intersection *isect = NULL, Spectrum *T = NULL,
        Spectrum *Ls = NULL, float* rayWeight = NULL, int count = 1) const;
    Spectrum Transmittance(const Scene *scene, const RayDifferential &ray,
        const Sample *sample, RNG &rng, MemoryArena &arena) const;
private:
    // GPURenderer Private Data
    Sampler *sampler;
    Camera *camera;
    SurfaceIntegrator *surfaceIntegrator;
    VolumeIntegrator *volumeIntegrator;

};


// SamplerRendererTask Declarations
class GPURendererTask : public Task {
public:
    // GPURendererTask Public Methods
    GPURendererTask(const Scene *sc, Renderer *ren, Camera *c,
                      ProgressReporter &pr,
                        Sampler *ms, Sample *sam, int tn, int tc)
      : reporter(pr)
    {
        scene = sc; renderer = ren; camera = c; mainSampler = ms;
        origSample = sam; taskNum = tn; taskCount = tc;
    }
    void Run();
private:
    // GPURendererTask Private Data
    const Scene *scene;
    const Renderer *renderer;
    Camera *camera;
    Sampler *mainSampler;
    ProgressReporter &reporter;
    Sample *origSample;
    int taskNum, taskCount;
};



#endif // PBRT_RENDERERS_SAMPLERRENDERER_H
