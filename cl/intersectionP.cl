#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, int count, int size)
{
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= count) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;

    float4 v1, v2, v3, rayd, rayo;
    rayd = (float4)(dir[3*iGID], dir[3*iGID+1], dir[3*iGID+2], 0);
    rayo = (float4)(o[3*iGID], o[3*iGID+1], o[3*iGID+2], 0);

    tHit[iGID] = '0';
    float tMax = bounds[iGID*2+1];

    for ( int i = 0; i < size; i++){
       v1 = (float4)(vertex[9*i], vertex[9*i+1], vertex[9*i+2], 0);
       v2 = (float4)(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5], 0);
       v3 = (float4)(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8], 0);
       e1 = v2 - v1;
       e2 = v3 - v1;
       s1 = cross(rayd, e2);
       divisor = dot(s1, e1);
       if ( divisor == 0.0) continue; //potrebuju nejake pole boolu
       invDivisor = 1.0f/ divisor;

	// compute first barycentric coordinate
	d = rayo - v1;
	b1 = dot(d, s1) * invDivisor;
	if ( b1 < -1e-3f  || b1 > 1.+1e-3f) continue;

	// compute second barycentric coordinate
	s2 = cross(d, e1);
	b2 = dot(rayd, s2) * invDivisor;
	if ( b2 < -1e-3f || (b1 + b2) > 1.+1e-3f) continue;

	// Compute _t_ to intersection point
	t = dot(e2, s2) * invDivisor;
	if (t < bounds[iGID*2]) continue;
	if (tHit[iGID] != INFINITY && tHit[iGID] != NAN && t > tHit[iGID]) continue;
	tHit[iGID] = '1';
	tMax = t;

    }
}

__kernel void IntersectionRP (
const __global float* vertex, const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, int count, int size)
{
    int iGID = get_global_id(0);
    if (iGID >= size) return;

    // process all geometry
    float4 e1, e2, s1, s2, d;
    float divisor, invDivisor, b1, b2, t;

    float4 v1, v2, v3, rayd, rayo;
    v1 = (float4)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2], 0);
    v2 = (float4)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5], 0);
    v3 = (float4)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8], 0);
    e1 = v2 - v1;
    e2 = v3 - v1;

    for ( int i = 0; i < count; i++){
       if ( tHit[i] == '1') continue; //already know it is occluded
       rayd = (float4)(dir[3*i], dir[3*i+1], dir[3*i+2], 0);
       rayo = (float4)(o[3*i], o[3*i+1], o[3*i+2], 0);
       s1 = cross(rayd, e2);
       divisor = dot(s1, e1);
       if ( divisor == 0.0) continue;
       invDivisor = 1.0f/ divisor;

	// compute first barycentric coordinate
	d = rayo - v1;
	b1 = dot(d, s1) * invDivisor;
	if ( b1 < -1e-3f  || b1 > 1.+1e-3f) continue;

	// compute second barycentric coordinate
	s2 = cross(d, e1);
	b2 = dot(rayd, s2) * invDivisor;
	if ( b2 < -1e-3f || (b1 + b2) > 1.+1e-3f) continue;

	// Compute _t_ to intersection point
	t = dot(e2, s2) * invDivisor;
	if (t < bounds[i*2]) continue;
	if (tHit[i] != INFINITY && tHit[i] != NAN && t > tHit[i]) continue;
	tHit[i] = '1';
     }
}

