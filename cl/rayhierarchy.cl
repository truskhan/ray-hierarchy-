#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002f

void intersectPAllLeaves (const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, float3 v1, float3 v2, float3 v3, float3 e1, float3 e2, int chunk, int rindex
){
    float3 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone
    for ( int i = 0; i < chunk; i++){
      rayd = (float3)(dir[3*rindex + 3*i], dir[3*rindex + 3*i+1], dir[3*rindex + 3*i+2]);
      rayo = (float3)(o[3*rindex + 3*i], o[3*rindex +3*i+1], o[3*rindex + 3*i+2]);
      s1 = cross(rayd, e2);
      divisor = dot(s1, e1);
      if ( divisor == 0.0f) continue;
      invDivisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      d = rayo - v1;
      b1 = dot(d, s1) * invDivisor;
      if ( b1 < -1e-3f  || b1 > 1+1e-3f) continue;

      // compute second barycentric coordinate
      s2 = cross(d, e1);
      b2 = dot(rayd, s2) * invDivisor;
      if ( b2 < -1e-3f || (b1 + b2) > 1+1e-3f) continue;

      // Compute _t_ to intersection point
      t = dot(e2, s2) * invDivisor;
      if (t < bounds[2*rindex + i*2]) continue;

      tHit[rindex+i] = '1';
    }
}

void intersectAllLeaves (const __global float* dir, const __global float* o,
const __global float* bounds, __global int* index, __global float* tHit, float3 v1, float3 v2, float3 v3,
float3 e1, float3 e2, int chunk, int rindex
#ifdef STAT_RAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
 ){
    float3 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone

    for ( int i = 0; i < chunk; i++){
      rayd = (float3)(dir[3*rindex + 3*i], dir[3*rindex + 3*i+1], dir[3*rindex + 3*i+2]);
      rayo = (float3)(o[3*rindex + 3*i], o[3*rindex +3*i+1], o[3*rindex + 3*i+2]);
      #ifdef STAT_RAY_TRIANGLE
       ++stat_rayTriangle[rindex + i];
      #endif
      s1 = cross(rayd, e2);
      divisor = dot(s1, e1);
      if ( divisor == 0.0f) continue;
      invDivisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      d = rayo - v1;
      b1 = dot(d, s1) * invDivisor;
      if ( b1 < -1e-3f  || b1 > 1+1e-3f) continue;

      // compute second barycentric coordinate
      s2 = cross(d, e1);
      b2 = dot(rayd, s2) * invDivisor;
      if ( b2 < -1e-3f || (b1 + b2) > 1+1e-3f) continue;

      // Compute _t_ to intersection point
      t = dot(e2, s2) * invDivisor;
      if (t < bounds[2*rindex + i*2]) continue;

      if (t > tHit[rindex + i]) continue;
        tHit[rindex + i] = t;
        index[rindex + i] = get_global_id(0);


    }

}

void yetAnotherIntersectAllLeaves (const __global float* dir, const __global float* o,
const __global float* bounds, __global int* index, __global float* tHit, __global int* changed,
float3 v1, float3 v2, float3 v3,
float3 e1, float3 e2, int chunk, int rindex ){
    float3 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone

    for ( int i = 0; i < chunk; i++){
      rayd = (float3)(dir[3*rindex + 3*i], dir[3*rindex + 3*i+1], dir[3*rindex + 3*i+2]);
      rayo = (float3)(o[3*rindex + 3*i], o[3*rindex +3*i+1], o[3*rindex + 3*i+2]);

      s1 = cross(rayd, e2);
      divisor = dot(s1, e1);
      if ( divisor == 0.0f) continue;
      invDivisor = 1.0f/ divisor;

      // compute first barycentric coordinate
      d = rayo - v1;
      b1 = dot(d, s1) * invDivisor;
      if ( b1 < -1e-3f  || b1 > 1+1e-3f) continue;

      // compute second barycentric coordinate
      s2 = cross(d, e1);
      b2 = dot(rayd, s2) * invDivisor;
      if ( b2 < -1e-3f || (b1 + b2) > 1+1e-3f) continue;

      // Compute _t_ to intersection point
      t = dot(e2, s2) * invDivisor;
      if (t < bounds[2*rindex + i*2]) continue;

      if (t > tHit[rindex + i]) continue;
        tHit[rindex + i] = t;
        index[rindex + i] = get_global_id(0);
        changed[get_global_id(0)] = 1;
    }

}

__kernel void computeDpTuTv (const __global float* vertex, const __global float* dir, const __global float* o,
                             const __global int* index, const __global float* uvs,
                             __global float* tutv, __global float* dp, int count ){

    int iGID = get_global_id(0);
    if ( iGID >= count ) return;
    int i = index[iGID];
    if ( i == 0 ) return;

    float3 rayd,rayo, v1, v2, v3, e1, e2;
    float b1,b2,invDivisor;

    rayd = (float3)(dir[3*iGID], dir[3*iGID+1], dir[3*iGID+2]);
    rayo = (float3)(o[3*iGID], o[3*iGID+1], o[3*iGID+2]);

    v1 = (float3)(vertex[9*i], vertex[9*i+1], vertex[9*i+2]);
    v2 = (float3)(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5]);
    v3 = (float3)(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8]);
    e1 = v2 - v1;
    e2 = v3 - v1;

    float3 s1 = cross(rayd, e2);
    float divisor = dot(s1, e1);
    invDivisor = 1.0f/ divisor;

   // compute first barycentric coordinate
    float3 d = rayo - v1;
    b1 = dot(d, s1) * invDivisor;

    // compute second barycentric coordinate
    float3 s2 = cross(d, e1);
    b2 = dot(rayd, s2) * invDivisor;

    float du1 = uvs[6*i]   - uvs[6*i+4];
    float du2 = uvs[6*i+2] - uvs[6*i+4];
    float dv1 = uvs[6*i+1] - uvs[6*i+5];
    float dv2 = uvs[6*i+3] - uvs[6*i+5];
    float3 dp1 = v1 - v3;
    float3 dp2 = v2 - v3;

    float determinant = du1 * dv2 - dv1 * du2;

    if ( determinant == 0.f ) {
      float3 temp = normalize(cross(e2, e1));
      if ( fabs(temp.x) > fabs(temp.y)) {
          float invLen = rsqrt(temp.x*temp.x + temp.z*temp.z);
          dp[6*iGID] = -temp.z*invLen;
          dp[6*iGID+1] = 0.f;
          dp[6*iGID+2] = temp.x*invLen;
      } else {
          float invLen = rsqrt(temp.y*temp.y + temp.z*temp.z);
          dp[6*iGID] = 0.f;
          dp[6*iGID+1] = temp.z*invLen;
          dp[6*iGID+2] = -temp.y*invLen;
      }
      float3 help = cross(temp, (float3)(dp[6*iGID], dp[6*iGID+1], dp[6*iGID+2]));
      dp[6*iGID+3] = help.x;
      dp[6*iGID+4] = help.y;
      dp[6*iGID+5] = help.z;
    } else {
      float invdet = 1.f / determinant;
      float3 help = (dv2 * dp1 - dv1 * dp2) * invdet;

      dp[6*iGID] = help.x;
      dp[6*iGID+1] = help.y;
      dp[6*iGID+2] = help.z;
      help = (-du2 * dp1 + du1 * dp2) * invdet;
      dp[6*iGID+3] = help.x;
      dp[6*iGID+4] = help.y;
      dp[6*iGID+5] = help.z;
  }

  float b0 = 1 - b1 - b2;
  tutv[2*iGID] = b0*uvs[6*i] + b1*uvs[6*i+2] + b2*uvs[6*i+4];
  tutv[2*iGID+1] = b0*uvs[6*i+1] + b1*uvs[6*i+3] + b2*uvs[6*i+5];

}

__kernel void levelConstruct(__global float* cones, const int count,
  const int threadsCount, const int level){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = threadsCount; //end of level0
  int temp;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;

  if ( iGID >= levelcount ) return;

  float3 x, q, c, a, g, xb, ab,e,n;
  float cosfi, sinfi, cosfib;
  float dotrx, dotcx, t ;
  float fi,fib;

  x = (float3)(cones[8*beginr + 16*iGID+3],cones[8*beginr + 16*iGID+4],cones[8*beginr + 16*iGID+5]);
  a = (float3)(cones[8*beginr + 16*iGID],cones[8*beginr + 16*iGID+1],cones[8*beginr + 16*iGID+2]);
  fi = cones[8*beginr + 16*iGID+6];
  cosfi = native_cos(fi);
  cones[8*beginw + 8*iGID + 7] = 1;
  if ( !( iGID == (levelcount - 1) && temp % 2 == 1 )) {
    //posledni vlakno jen prekopiruje
    cones[8*beginw + 8*iGID + 7] = 2;
    ab = (float3)(cones[8*beginr + 16*iGID+8],cones[8*beginr + 16*iGID+9],cones[8*beginr + 16*iGID+10]);
    xb = (float3)(cones[8*beginr + 16*iGID+11],cones[8*beginr + 16*iGID+12],cones[8*beginr + 16*iGID+13]);
    fib = cones[8*beginr + 16*iGID+13];
    cosfib = native_cos(fib);

    //average direction
    dotrx = dot(x,xb);
    if ( dotrx < cosfib && dotrx < cosfi){
      x = (x+xb)/length(x+xb);
      cosfi = native_cos(acos(dotrx) + min(fib,fi));
      fi = acos(cosfi);
    } else {
      if ( fi < fib){
        x = xb;
        a = ab;
        cosfi = cosfib;
        fi = fib;
      }
    }
    //move the apex
    c = ab - a;
    if ( length(c) > EPS){
      c = normalize(c);
      dotcx = dot(x,c);
      if ( dotcx < cosfi){
        q = (dotcx*x - c)/length(dotcx*x-x);
        sinfi = native_sin(fi);
        e = x*cosfi + q*sinfi;
        n = x*cosfi - q*sinfi;
        g = c - dot(n,c)*n;
        t = (length(g)*length(g))/dot(e,g);
        a = a - t*e;
      }
    }
  }
    cones[8*beginw + 8*iGID]     = a.x;
    cones[8*beginw + 8*iGID + 1] = a.y;
    cones[8*beginw + 8*iGID + 2] = a.z;
    cones[8*beginw + 8*iGID + 3] = x.x;
    cones[8*beginw + 8*iGID + 4] = x.y;
    cones[8*beginw + 8*iGID + 5] = x.z;
    cones[8*beginw + 8*iGID + 6] = fi;
}

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
  const __global unsigned int* counts, __global float* cones, const int count){
  int iGID = get_global_id(0);
  if (iGID >= count ) return;

  float3 x, r, q, c, p , a, e, n , g;
  float cosfi, sinfi;
  float dotrx, dotcx, t ;

  unsigned int index = 0;
  for ( int i = 0; i < iGID; i++)
    index += counts[i];

  //start with the zero angle enclosing cone of the first ray
  x = normalize((float3)(dir[3*index],dir[3*index+1],dir[3*index+2]));
  a = (float3)(o[3*index],o[3*index+1], o[3*index+2]);
  cosfi = 1;

  for ( int i = 1; i < counts[iGID]; i++){
    //check if the direction of the ray lies within the solid angle
    r = normalize((float3)(dir[3*(index+i)], dir[3*(index+i)+1],dir[3*(index+i)+2]));
    p = (float3)(o[3*(index+i)], o[3*(index+i)+1], o[3*(index+i)+2]);
    dotrx = dot(r,x);
    if ( dotrx < cosfi ){
      //extend the cone
      q = normalize(dotrx*x - r);
      sinfi = (cosfi > (1-EPS))? 0:native_sin(acos(cosfi)); //precison problems
      e = normalize(x*cosfi + q*sinfi);
      x = normalize(e+r);

      cosfi = dot(x,r);
    }
    //check if the origin of the ray is within the wolume
    if ( length(p-a) > EPS){
      c = normalize(p - a);
      dotcx = dot(c,x);
      if ( dotcx < cosfi){
        q = (dotcx*x - c)/length(dotcx*x-x);
        sinfi = native_sin(acos(cosfi));
        e = x*cosfi + q*sinfi;
        n = x*cosfi - q*sinfi;
        g = c - dot(n,c)*n;
        t = (length(g)*length(g))/dot(e,g);
        a = a - t*e;
      }
    }
  }

  //store the result
  cones[8*iGID]   = a.x;
  cones[8*iGID+1] = a.y;
  cones[8*iGID+2] = a.z;
  cones[8*iGID+3] = x.x;
  cones[8*iGID+4] = x.y;
  cones[8*iGID+5] = x.z;
  cones[8*iGID+6] = (cosfi > (1-EPS)) ? 0.003f: acos(cosfi); //precision problems
  cones[8*iGID+7] = counts[iGID];

}

int computeChild (unsigned int threadsCount, int i){
  int index = 0;
  int levelcount = threadsCount;
  int temp;

  if ( i < 8*levelcount)
    return -1; // level 0, check rays

  while ( (index + 8*levelcount) <= i){
    temp = levelcount;
    index += 8*levelcount;
    levelcount = (levelcount+1)/2;
  }
  int offset = i - index;

  return (index - 8*temp) + 2*offset;
}

int computeRIndex (unsigned int j, const __global float* cones){
  int rindex = 0;
  for ( int i = 0; i < j; i += 8){
      rindex += cones[i + 7];
  }
  return rindex;
}

__kernel void IntersectionR (
    const __global float* vertex, const __global float* dir, const __global float* o,
    const __global float* cones, const __global float* bounds, __global float* tHit,
    __global int* index,
#ifdef STAT_TRIANGLE_CONE
 __global int* stat_triangleCone,
#endif
#ifdef STAT_RAY_TRIANGLE
 __global int* stat_rayTriangle,
#endif
    __local int* stack,
     int count, int size, int chunk, int height, unsigned int threadsCount
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    #ifdef STAT_TRIANGLE_CONE
    stat_triangleCone[iGID] = 0;
    #endif

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;

    // find geometry for the work-item
    float3 e1, e2;

    float3 v1, v2, v3;
    v1 = (float3)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2]);
    v2 = (float3)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5]);
    v3 = (float3)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8]);
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding sphere
    //vizualizace pruseciku s paprskem a kuzel,trojuhelnikem, ktery se pocitaly

    float3 center; float radius;
    //bounding sphere center - center of mass
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float3 a,x;
    float fi;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin, rindex;
    int i = 0;
    int child;
    float len;

    begin = 8*num;
    for ( int j = 0; j < levelcount; j++){
      // get cone description
      a = (float3)(cones[begin + 8*j],cones[begin + 8*j+1],cones[begin + 8*j+2]);
      x = (float3)(cones[begin + 8*j+3],cones[begin + 8*j+4],cones[begin + 8*j+5]);
      fi = cones[begin + 8*j+6];
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        #ifdef STAT_TRIANGLE_CONE
         ++stat_triangleCone[iGID];
        #endif
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 8*lastlevelnum + 16*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          a = (float3)(cones[i],cones[i+1],cones[i+2]);
          x = (float3)(cones[i+3],cones[i+4],cones[i+5]);
          fi = cones[i+6];
          len = length(center-a);
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            #ifdef STAT_PICTURE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,cones[i+7], rindex
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
            }
          }
          a = (float3)(cones[i+8],cones[i+9],cones[i+10]);
          x = (float3)(cones[i+11],cones[i+12],cones[i+13]);
          fi = cones[i+14];
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
         {
            #ifdef STAT_TRIANGLE_CONE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild (threadsCount, i+8);
            //if the cone is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i + 8, cones);
              intersectAllLeaves( dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,cones[i+15],rindex
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            }
            else {
              stack[iLID*(height) + SPindex++] = child;
            }
          }
        }
      }

    }


}

__kernel void YetAnotherIntersection (
    const __global float* vertex, const __global float* dir, const __global float* o,
    const __global float* cones, const __global float* bounds, __global float* tHit,
    __global int* index, __global int* changed,
    __local int* stack,
     int count, int size, int chunk, int height, unsigned int threadsCount
) {
    // find position in global and shared arrays
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;

    // find geometry for the work-item
    float3 e1, e2;

    float3 v1, v2, v3;
    v1 = (float3)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2]);
    v2 = (float3)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5]);
    v3 = (float3)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8]);
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding sphere
    float3 center; float radius;
    //bounding sphere center - center of mass
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float3 a,x;
    float fi;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin, rindex;
    int i = 0;
    int child;
    float len;

    begin = 8*num;
    for ( int j = 0; j < levelcount; j++){
      // get cone description
      a = (float3)(cones[begin + 8*j],cones[begin + 8*j+1],cones[begin + 8*j+2]);
      x = (float3)(cones[begin + 8*j+3],cones[begin + 8*j+4],cones[begin + 8*j+5]);
      fi = cones[begin + 8*j+6];
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 8*lastlevelnum + 16*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          a = (float3)(cones[i],cones[i+1],cones[i+2]);
          x = (float3)(cones[i+3],cones[i+4],cones[i+5]);
          fi = cones[i+6];
          len = length(center-a);
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            child = computeChild(threadsCount,i);
            //if the cones is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i, cones);
              yetAnotherIntersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,cones[i+7], rindex);
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
            }
          }
          a = (float3)(cones[i+8],cones[i+9],cones[i+10]);
          x = (float3)(cones[i+11],cones[i+12],cones[i+13]);
          fi = cones[i+14];
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
         {
            child = computeChild (threadsCount, i+8);
            //if the cone is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i + 8, cones);
              yetAnotherIntersectAllLeaves( dir, o, bounds, index, tHit, changed, v1,v2,v3,e1,e2,cones[i+15],rindex);
            }
            else {
              stack[iLID*(height) + SPindex++] = child;
            }
          }
        }
      }

    }


}



__kernel void IntersectionP (
const __global float* vertex, const __global float* dir, const __global float* o,
 const __global float* cones, const __global float* bounds,
__global unsigned char* tHit, __local int* stack, int count, int size, int chunk, int height,unsigned int threadsCount)
{
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    if (iGID >= size) return;

    // process all geometry
    float3 e1, e2;

    float3 v1, v2, v3;
    v1 = (float3)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2]);
    v2 = (float3)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5]);
    v3 = (float3)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8]);
    e1 = v2 - v1;
    e2 = v3 - v1;

    float3 center; float radius;
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float3 a,x;
    float fi;
    float len;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = threadsCount; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin,rindex;
    int i = 0;
    int child;

    begin = 8*num;
    for ( int j = 0; j < levelcount; j++){
      // get cone description
      a = (float3)(cones[begin + 8*j],cones[begin + 8*j+1],cones[begin + 8*j+2]);
      x = (float3)(cones[begin + 8*j+3],cones[begin + 8*j+4],cones[begin + 8*j+5]);
      fi = cones[begin + 8*j+6];
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        //store child to the stack
        stack[iLID*height + SPindex++] = begin - 8*lastlevelnum + 16*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*height + SPindex];
          a = (float3)(cones[i],cones[i+1],cones[i+2]);
          x = (float3)(cones[i+3],cones[i+4],cones[i+5]);
          fi = cones[i+6];
          len = length(center-a);
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            child = computeChild (threadsCount, i);
            //if the cones is at level 0 - check leaves
            if ( child < 0){
              rindex = computeRIndex(i,cones);
              intersectPAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,cones[7],rindex);
            }
            else {
              //save the intersected cone to the stack
              stack[iLID*height + SPindex++] = child;
            }
          }
          a = (float3)(cones[i+8],cones[i+9],cones[i+10]);
          x = (float3)(cones[i+11],cones[i+12],cones[i+13]);
          fi = cones[i+14];
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
         {
            child = computeChild (threadsCount, i+8);
            //if the cone is at level 0 - check leaves
            if ( child < 0) {
              rindex = computeRIndex(i + 8, cones);
              intersectPAllLeaves( dir, o, bounds, tHit, v1,v2,v3,e1,e2,cones[i+15],rindex);
            }
            else {
              stack[iLID*height + SPindex++] = child;
            }
          }
        }
      }

    }

}

