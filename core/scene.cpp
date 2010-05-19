
/*
    pbrt source code Copyright(c) 1998-2009 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    pbrt is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.  Note that the text contents of
    the book "Physically Based Rendering" are *not* licensed under the
    GNU GPL.

    pbrt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */


// core/scene.cpp*
#include "scene.h"
#include "camera.h"
#include "film.h"
#include "sampler.h"
#include "volume.h"
#include "parallel.h"
#include "progressreporter.h"
#include "renderer.h"

// Scene Method Definitions
Scene::~Scene() {
    delete aggregate;
    delete volumeRegion;
    for (uint32_t i = 0; i < lights.size(); ++i)
        delete lights[i];
}


Scene::Scene(Primitive *accel, const vector<Light *> &lts,
             VolumeRegion *vr) {
    lights = lts;
    aggregate = accel;
    volumeRegion = vr;
    if (lts.size() == 0)
        Warning("No light sources defined in scene; "
            "possibly rendering a black image.");
    // Scene Constructor Implementation
    bound = aggregate->WorldBound();
    if (volumeRegion) bound = Union(bound, volumeRegion->WorldBound());
}

unsigned int Scene::MaxRaysPerCall() const {
    RayHieararchy* rh = dynamic_cast<RayHieararchy*>(aggregate);
    if ( rh != NULL)
        return rh->MaxRaysPerCall();
    else
      Severe("Called MaxRaysPerCall with unsoppurted aggregate");
}

void Scene::Intersect(const RayDifferential* ray, Intersection *isect, bool* hit, float* rayWeight,int & count
#ifdef STAT_RAY_TRIANGLE
, Spectrum *Ls
#endif
) const {
    NaiveAccel* na = dynamic_cast<NaiveAccel*> (aggregate);
    if ( na != NULL)
      na->Intersect(ray, isect, rayWeight, hit, count);
    else {
      RayHieararchy* rh = dynamic_cast<RayHieararchy*>(aggregate);
      if ( rh != NULL)
        rh->Intersect(ray, isect, rayWeight, hit,count
        #ifdef STAT_RAY_TRIANGLE
        , Ls
        #endif
        );
      else
        Severe("Called Intersect with unsoppurted aggregate");
    }

}

const BBox &Scene::WorldBound() const {
    return bound;
}


