


// renderers/breadthFirst.cpp*
#include "renderers/breadth-first.h"
#include "scene.h"
#include "film.h"
#include "volume.h"
#include "sampler.h"
#include "integrator.h"
#include "progressreporter.h"
#include "camera.h"
#include "intersection.h"

// breadthFirstTask Definitions
void breadthFirstTask::Run() {
    PBRT_STARTED_RENDERTASK(taskNum);
    // Get sub-_Sampler_ for _SamplerRendererTask_
    Sampler *sampler = mainSampler; //->GetSubSampler(taskNum, taskCount);
    if (!sampler)
    {
        reporter.Update();
        PBRT_FINISHED_RENDERTASK(taskNum);
        return;
    }

    // Declare local variables used for rendering loop
    MemoryArena arena;
    RNG rng(taskNum);

    // Allocate space for samples and intersections
    int maxSamples = (camera->film->xResolution * camera->film->yResolution);
    Sample *samples = origSample->Duplicate(maxSamples);

    RayDifferential *rays = new RayDifferential[maxSamples];
    Spectrum *Ls = new Spectrum[maxSamples];
    Spectrum *Ts = new Spectrum[maxSamples];
    Intersection *isects = new Intersection[maxSamples];

    // Get samples from \use{Sampler} and update image
    int sampleCount = 0;
    int counter = 0;
    int temp = 0;
    // predpokladam jeden sample na pixel, jinak by byla race condition
    while ((counter != maxSamples) && (temp = sampler->GetMoreSamples(&samples[counter],rng)>0)){
      ++counter;
      sampleCount += temp; //we just take one sample per pixel, so it could be +1
    }

    // Generate camera rays
    float* rayWeight = new float[maxSamples];
    for (int i = 0; i < sampleCount; ++i){
        // Find camera ray for _sample[i]_
        PBRT_STARTED_GENERATING_CAMERA_RAY(&samples[i]);
        rayWeight[i] = camera->GenerateRayDifferential(samples[i], &rays[i]);
        PBRT_FINISHED_GENERATING_CAMERA_RAY(&samples[i], &rays[i], rayWeight);
    }

    // Evaluate radiance along camera rays
    PBRT_STARTED_CAMERA_RAY_INTEGRATION(&rays[i], &samples[i]);
    renderer->Li(scene, rays, samples, rng, arena, isects, Ts, Ls, rayWeight, sampleCount);
    PBRT_FINISHED_CAMERA_RAY_INTEGRATION(&rays[i], &samples[i], &Ls[i]);

    // Report sample results to _Sampler_, add contributions to image
    if (sampler->ReportResults(samples, rays, Ls, isects, sampleCount))
    {
        for (int i = 0; i < sampleCount; ++i)
        {
            PBRT_STARTED_ADDING_IMAGE_SAMPLE(&samples[i], &rays[i], &Ls[i], &Ts[i]);
            camera->film->AddSample(samples[i], Ls[i]);
            PBRT_FINISHED_ADDING_IMAGE_SAMPLE();
        }
    }

    // Free \use{MemoryArena} memory from computing image sample values
    arena.FreeAll();


    // Clean up after \use{breadthFirstTask} is done with its image region
    camera->film->UpdateDisplay(sampler->xPixelStart,
        sampler->yPixelStart, sampler->xPixelEnd+1, sampler->yPixelEnd+1);
    //use of global sampler, don't delete it, it will be done by ~breadthFirst
    //delete sampler;
    delete[] samples;
    delete[] rays;
    delete[] Ls;
    delete[] Ts;
    delete[] isects;
    delete[] rayWeight;
    reporter.Update();
    PBRT_FINISHED_RENDERTASK(taskNum);
}



// breadthFirst Method Definitions
breadthFirst::breadthFirst(Sampler *s, Camera *c,
        SurfaceIntegrator *si, VolumeIntegrator *vi) {
    sampler = s;
    camera = c;
    surfaceIntegrator = si;
    volumeIntegrator = vi;
}


breadthFirst::~breadthFirst() {
    delete sampler;
    delete camera;
    delete surfaceIntegrator;
    delete volumeIntegrator;
}


void breadthFirst::Render(const Scene *scene) {
    PBRT_FINISHED_PARSING();
    // Allow integrators to do pre-processing for the scene
    PBRT_STARTED_PREPROCESSING();
    surfaceIntegrator->Preprocess(scene, camera, this);
    volumeIntegrator->Preprocess(scene, camera, this);
    PBRT_FINISHED_PREPROCESSING();
    PBRT_STARTED_RENDERING();
    // Allocate and initialize _sample_
    Sample *sample = new Sample(sampler, surfaceIntegrator,
                                volumeIntegrator, scene);

    // Create and launch _SamplerRendererTask_s for rendering image

    // Compute number of _SamplerRendererTask_s to create for rendering
    //int nPixels = camera->film->xResolution * camera->film->yResolution;
    //int nTasks = max(32 * NumSystemCores(), nPixels / (16*16));
    int nTasks = sampler->samplesPerPixel;
    nTasks = RoundUpPow2(nTasks);
    ProgressReporter reporter(nTasks, "Rendering");
    vector<Task *> renderTasks;
    /*for (int i = 0; i < nTasks; ++i)
        renderTasks.push_back(new breadthFirstTask(scene, this, camera,
                                  reporter,
                                    sampler, sample, nTasks-1-i, nTasks));
    EnqueueTasks(renderTasks);
    WaitForAllTasks();*/
    breadthFirstTask(scene, this, camera,
                                  reporter,
                                    sampler, sample, nTasks-1, nTasks).Run();
    for (uint32_t i = 0; i < renderTasks.size(); ++i)
        delete renderTasks[i];
    reporter.Done();
    PBRT_FINISHED_RENDERING();
    // Clean up after rendering and store final image
    delete sample;
    camera->film->WriteImage();
}

void breadthFirst::Li(const Scene *scene, const RayDifferential* ray,
        const Sample *sample, RNG &rng, MemoryArena &arena,
        Intersection *isect, Spectrum *T , Spectrum *Ls, float* rayWeight, int count ) const {
  bool* hit = new bool[count];
  scene->Intersect(ray, isect, hit, rayWeight, count);
  Spectrum* Lo = new Spectrum[count];
  for ( int i = 0; i < count; i++)
    Lo[i] = Spectrum(0.f);
  surfaceIntegrator->Li(scene, this, ray, isect, sample, rng, arena, rayWeight, Lo, hit, count);
  for ( int i = 0; i < count; i++){
    if (rayWeight[i] <= 0.f){
        Ls[i] = 0.f;
        T[i] = 1.f;
        continue;
    }
    Assert(ray[i].time == sample[i].time);
    Assert(!ray[i].HasNaNs());

    if (!hit[i])  {
        // Handle ray that doesn't intersect any geometry
        for (uint32_t j = 0; j < scene->lights.size(); ++j)
           Lo[i] += scene->lights[j]->Le(ray[i]);
    }
    Spectrum Lv = volumeIntegrator->Li(scene, this, ray[i], &sample[i], rng,
                                       &T[i], arena);
    Ls[i] = T[i] * Lo[i] + Lv;
   }
   delete [] hit;
   delete [] Lo;
}

Spectrum breadthFirst::Li(const Scene *scene,
        const RayDifferential &ray, const Sample *sample, RNG &rng,
        MemoryArena &arena, Intersection *isect, Spectrum *T) const {
    Assert(ray.time == sample->time);
    Assert(!ray.HasNaNs());
    // Allocate local variables for _isect_ and _T_ if needed
    Spectrum localT;
    if (!T) T = &localT;
    Intersection localIsect;
    if (!isect) isect = &localIsect;
    Spectrum Lo = 0.f;
    if (scene->Intersect(ray, isect))
        Lo = surfaceIntegrator->Li(scene, this, ray, *isect, sample,
                                   rng, arena);
    else {
        // Handle ray that doesn't intersect any geometry
        for (uint32_t i = 0; i < scene->lights.size(); ++i)
           Lo += scene->lights[i]->Le(ray);
    }
    Spectrum Lv = volumeIntegrator->Li(scene, this, ray, sample, rng,
                                       T, arena);
    return *T * Lo + Lv;
}


Spectrum breadthFirst::Transmittance(const Scene *scene,
        const RayDifferential &ray, const Sample *sample, RNG &rng,
        MemoryArena &arena) const {
    return volumeIntegrator->Transmittance(scene, this, ray, sample,
                                           rng, arena);
}


