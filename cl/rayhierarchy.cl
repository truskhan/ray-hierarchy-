#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#define EPS 0.000002

void intersectPAllLeaves (int j, const __global float* dir, const __global float* o, const __global float* bounds,
__global unsigned char* tHit, float4 v1, float4 v2, float4 v3, float4 e1, float4 e2, int chunk
){
    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone
    for ( int i = 0; i < chunk; i++){
      rayd = (float4)(dir[3*j*chunk + 3*i], dir[3*j*chunk + 3*i+1], dir[3*j*chunk + 3*i+2], 0);
      rayo = (float4)(o[3*j*chunk + 3*i], o[3*j*chunk +3*i+1], o[3*j*chunk + 3*i+2], 0);
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
      if (t < bounds[2*j*chunk + i*2]) continue;

      //if (tHit[j*chunk+i] != INFINITY && tHit[j*chunk + i] != NAN && t > tHit[j*chunk + i]) continue;
      tHit[j*chunk+i] = '1';
    }
}

void intersectAllLeaves (int j, const __global float* dir, const __global float* o,
const __global float* bounds, __global int* index, __global float* tHit, float4 v1, float4 v2, float4 v3,
float4 e1, float4 e2, int chunk, int count
#ifdef STAT_RAY_TRIANGLE
, __global int* stat_rayTriangle
#endif
 ){
    float4 s1, s2, d, rayd, rayo;
    float divisor, invDivisor, t, b1, b2;
    // process all rays in the cone

    for ( int i = 0; i < chunk; i++){
      rayd = (float4)(dir[3*j*chunk + 3*i], dir[3*j*chunk + 3*i+1], dir[3*j*chunk + 3*i+2], 0);
      rayo = (float4)(o[3*j*chunk + 3*i], o[3*j*chunk +3*i+1], o[3*j*chunk + 3*i+2], 0);
      #ifdef STAT_RAY_TRIANGLE
       ++stat_rayTriangle[j*chunk + i];
      #endif
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
      if (t < bounds[2*j*chunk + i*2]) continue;
      //if ( j*chunk + i >= count ) continue; //proc se tohle stava?

      if (t > tHit[j*chunk + i]) continue;
      /*#ifdef STAT_RAY_TRIANGLE
       stat_rayTriangle[j*chunk + i] += 100;
      #endif*/
        tHit[j*chunk + i] = t;
        index[j*chunk + i] = get_global_id(0);


    }

}

__kernel void computeDpTuTv (const __global float* vertex, const __global float* dir, const __global float* o,
                             const __global int* index, const __global float* uvs,
                             __global float* tutv, __global float* dp, int count ){

    int iGID = get_global_id(0);
    if ( iGID >= count ) return;
    int i = index[iGID];
    if ( i == 0 ) return;

    float4 rayd,rayo, v1, v2, v3, e1, e2;
    float b1,b2,invDivisor;

    rayd = (float4)(dir[3*iGID], dir[3*iGID+1], dir[3*iGID+2], 0);
    rayo = (float4)(o[3*iGID], o[3*iGID+1], o[3*iGID+2], 0);

    v1 = (float4)(vertex[9*i], vertex[9*i+1], vertex[9*i+2], 0);
    v2 = (float4)(vertex[9*i + 3], vertex[9*i + 4], vertex[9*i + 5], 0);
    v3 = (float4)(vertex[9*i + 6], vertex[9*i + 7], vertex[9*i + 8], 0);
    e1 = v2 - v1;
    e2 = v3 - v1;

    float4 s1 = cross(rayd, e2);
    float divisor = dot(s1, e1);
    invDivisor = 1.0f/ divisor;

   // compute first barycentric coordinate
    float4 d = rayo - v1;
    b1 = dot(d, s1) * invDivisor;

    // compute second barycentric coordinate
    float4 s2 = cross(d, e1);
    b2 = dot(rayd, s2) * invDivisor;

    float du1 = uvs[6*i]   - uvs[6*i+4];
    float du2 = uvs[6*i+2] - uvs[6*i+4];
    float dv1 = uvs[6*i+1] - uvs[6*i+5];
    float dv2 = uvs[6*i+3] - uvs[6*i+5];
    float4 dp1 = v1 - v3;
    float4 dp2 = v2 - v3;

    float determinant = du1 * dv2 - dv1 * du2;

    if ( determinant == 0.f ) {
      float4 temp = normalize(cross(e2, e1));
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
      float4 help = cross(temp, (float4)(dp[6*iGID], dp[6*iGID+1], dp[6*iGID+2],0));
      dp[6*iGID+3] = help.x;
      dp[6*iGID+4] = help.y;
      dp[6*iGID+5] = help.z;
    } else {
      float invdet = 1.f / determinant;
      float4 help = (dv2 * dp1 - dv1 * dp2) * invdet;

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

__kernel void rayLevelConstruct(__global float* cones, const int count, const int chunk, const int level){
  int iGID = get_global_id(0);

  int beginr = 0;
  int beginw = 0;
  int levelcount = (count+chunk-1)/chunk; //end of level0
  int temp;

  for ( int i = 0; i < level; i++){
      beginw += levelcount;
      temp = levelcount;
      levelcount = (levelcount+1)/2; //number of elements in level
  }
  beginr = beginw - temp;

  if ( iGID >= levelcount ) return;

  float4 x, r, q, c, a, g, xb, ab,e,n;
  float cosfi, sinfi, cosfib, sinfib;
  float dotrx, dotcx, t ;
  float fi,fib;

  x = (float4)(cones[7*beginr + 14*iGID+3],cones[7*beginr + 14*iGID+4],cones[7*beginr + 14*iGID+5],0);
  a = (float4)(cones[7*beginr + 14*iGID],cones[7*beginr + 14*iGID+1],cones[7*beginr + 14*iGID+2],0);
  fi = cones[7*beginr + 14*iGID+6];
  cosfi = native_cos(fi);
  if ( iGID != levelcount ) {
    //posledni vlakno jen prekopiruje
    ab = (float4)(cones[7*beginr + 14*iGID+7],cones[7*beginr + 14*iGID+8],cones[7*beginr + 14*iGID+9],0);
    xb = (float4)(cones[7*beginr + 14*iGID+10],cones[7*beginr + 14*iGID+11],cones[7*beginr + 14*iGID+12],0);
    fib = cones[7*beginr + 14*iGID+13];
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
    cones[7*beginw + 7*iGID]     = a.x;
    cones[7*beginw + 7*iGID + 1] = a.y;
    cones[7*beginw + 7*iGID + 2] = a.z;
    cones[7*beginw + 7*iGID + 3] = x.x;
    cones[7*beginw + 7*iGID + 4] = x.y;
    cones[7*beginw + 7*iGID + 5] = x.z;
    cones[7*beginw + 7*iGID + 6] = fi;

}

__kernel void rayhconstruct(const __global float* dir,const  __global float* o,
 __global float* cones, const int chunk, const int count){
  int iGID = get_global_id(0);
  int total = (count+chunk-1)/chunk;
  if (iGID >= total) return;

  float4 x, r, q, xnew, c, p , a, e, n , g, xb, ab, rb, test;
  float cosfi, sinfi, cosfib, sinfib;
  float dotrx, dotcx, t ;
  if ( iGID >= count) return; //should not happend
  //start with the zero angle enclosing cone of the first ray
  x = normalize((float4)(dir[3*iGID*chunk],dir[3*iGID*chunk+1],dir[3*iGID*chunk+2],0));
  a = (float4)(o[3*iGID*chunk],o[3*iGID*chunk+1], o[3*iGID*chunk+2],0);
  cosfi = 1;

  for ( int i = 1; i < chunk && iGID*chunk+i < count; i++){
    //check if the direction of the ray lies within the solid angle
    r = normalize((float4)(dir[3*(iGID*chunk+i)], dir[3*(iGID*chunk+i)+1],dir[3*(iGID*chunk+i)+2],0));
    p = (float4)(o[3*(iGID*chunk+i)], o[3*(iGID*chunk+i)+1], o[3*(iGID*chunk+i)+2],0);
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
  cones[7*iGID]   = a.x;
  cones[7*iGID+1] = a.y;
  cones[7*iGID+2] = a.z;
  cones[7*iGID+3] = x.x;
  cones[7*iGID+4] = x.y;
  cones[7*iGID+5] = x.z;
  cones[7*iGID+6] = (cosfi > (1-EPS)) ? 0.003: acos(cosfi); //precision problems

}

int computeChild (int count, int chunk, int i){
  int index = 0;
  int levelcount = (count+chunk-1)/chunk;
  int temp;

  if ( i < 7*levelcount)
    return -1; // level 0, check rays

  while ( (index + 7*levelcount) <= i){
    temp = levelcount;
    index += 7*levelcount;
    levelcount = (levelcount+1)/2;
  }
  int offset = i - index;

  return (index - 7*temp) + 2*offset;
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
     int count, int size, int chunk, int height
) {
    // find position in global arrays
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    #ifdef STAT_TRIANGLE_CONE
    stat_triangleCone[iGID] = 0;
    #endif

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= size) return;

    /*for ( int i = iGID; i < count; i+=size ){
      index[i] = 0; //initialize the array
    }*/

    // find geometry for the work-item
    float4 e1, e2, test;
    float  t;

    float4 v1, v2, v3, rayd, rayo;
    v1 = (float4)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2], 0);
    v2 = (float4)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5], 0);
    v3 = (float4)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8], 0);
    e1 = v2 - v1;
    e2 = v3 - v1;

    //calculate bounding sphere
    //vizualizace pruseciku s paprskem a kuzel,trojuhelnikem, ktery se pocitaly

    float4 center; float radius;
    //bounding sphere center - center of mass
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float4 a,x;
    float fi;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = (count+chunk-1)/chunk; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin;
    int i = 0;
    int child;
    float len;

    for ( int j = 0; j < levelcount; j++){
      // get cone description
      begin = 7*num;
      a = (float4)(cones[begin + 7*j],cones[begin + 7*j+1],cones[begin + 7*j+2],0);
      x = (float4)(cones[begin + 7*j+3],cones[begin + 7*j+4],cones[begin + 7*j+5],0);
      fi = cones[begin + 7*j+6];
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        #ifdef STAT_TRIANGLE_CONE
         ++stat_triangleCone[iGID];
        #endif
        //store child to the stack
        stack[iLID*(height) + SPindex++] = begin - 7*lastlevelnum + 14*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*(height) + SPindex];
          a = (float4)(cones[i],cones[i+1],cones[i+2],0);
          x = (float4)(cones[i+3],cones[i+4],cones[i+5],0);
          fi = cones[i+6];
          len = length(center-a);
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            #ifdef STAT_PICTURE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild (count, chunk, i);
            //if the cones is at level 0 - check leaves
            if ( child < 0)
              intersectAllLeaves(i/7, dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,chunk,count
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
            else {
              //save the intersected cone to the stack
              stack[iLID*(height) + SPindex++] = child;
            }
          }
          a = (float4)(cones[i+7],cones[i+8],cones[i+9],0);
          x = (float4)(cones[i+10],cones[i+11],cones[i+12],0);
          fi = cones[i+13];
          if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
         {
            #ifdef STAT_TRIANGLE_CONE
             ++stat_triangleCone[iGID];
            #endif
            child = computeChild (count, chunk, i+7);
            //if the cone is at level 0 - check leaves
            if ( child < 0)
              intersectAllLeaves(i/7+1, dir, o, bounds, index, tHit, v1,v2,v3,e1,e2,chunk,count
              #ifdef STAT_RAY_TRIANGLE
                , stat_rayTriangle
              #endif
              );
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
__global unsigned char* tHit, __local int* stack, int count, int size, int chunk, int height)
{
    int iGID = get_global_id(0);
    int iLID = get_local_id(0);
    if (iGID >= size) return;

    // process all geometry
    float4 e1, e2, s1, s2, d, test;
    float divisor, invDivisor, b1, b2, t;

    float4 v1, v2, v3, rayd, rayo;
    v1 = (float4)(vertex[9*iGID], vertex[9*iGID+1], vertex[9*iGID+2], 0);
    v2 = (float4)(vertex[9*iGID + 3], vertex[9*iGID + 4], vertex[9*iGID + 5], 0);
    v3 = (float4)(vertex[9*iGID + 6], vertex[9*iGID + 7], vertex[9*iGID + 8], 0);
    e1 = v2 - v1;
    e2 = v3 - v1;

    float4 center; float radius;
    center = (v1+v2+v3)/3;
    radius = length(v1-center);

    float4 a,x;
    float fi;
    float len;

    //find number of elements in top level of the ray hieararchy
    uint levelcount = (count+chunk-1)/chunk; //end of level0
    uint num = 0;
    uint lastlevelnum = 0;

    for ( int i = 1; i < height; i++){
        lastlevelnum = levelcount;
        num += levelcount;
        levelcount = (levelcount+1)/2; //number of elements in level
    }

    int SPindex = 0;
    uint begin;
    int i = 0;
    int child;

    for ( int j = 0; j < levelcount; j++){
      // get cone description
      begin = 7*num;
      a = (float4)(cones[begin + 7*j],cones[begin + 7*j+1],cones[begin + 7*j+2],0);
      x = (float4)(cones[begin + 7*j+3],cones[begin + 7*j+4],cones[begin + 7*j+5],0);
      fi = cones[begin + 7*j+6];
      // check if triangle intersects cone
      len = length(center-a);
      if ( acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
      {
        //store child to the stack
        stack[iLID*height + SPindex++] = begin - 7*lastlevelnum + 14*j;
        while ( SPindex > 0 ){
          //take the cones from the stack and check them
          --SPindex;
          i = stack[iLID*height + SPindex];
          a = (float4)(cones[i],cones[i+1],cones[i+2],0);
          x = (float4)(cones[i+3],cones[i+4],cones[i+5],0);
          fi = cones[i+6];
          len = length(center-a);
          //if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
          {
            child = computeChild (count, chunk, i);
            //if the cones is at level 0 - check leaves
            if ( child < 0)
              intersectPAllLeaves(i/7, dir, o, bounds, tHit, v1,v2,v3,e1,e2,chunk);
            else {
              //save the intersected cone to the stack
              stack[iLID*height + SPindex++] = child;
            }
          }
          a = (float4)(cones[i+7],cones[i+8],cones[i+9],0);
          x = (float4)(cones[i+10],cones[i+11],cones[i+12],0);
          fi = cones[i+13];
         // if ( len < EPS || acos(dot((center-a)/len,x)) - asin(radius/len) < fi)
         {
            child = computeChild (count, chunk, i+7);
            //if the cone is at level 0 - check leaves
            if ( child < 0)
              intersectPAllLeaves(i/7+1, dir, o, bounds, tHit, v1,v2,v3,e1,e2,chunk);
            else {
              stack[iLID*height + SPindex++] = child;
            }
          }
        }
      }

    }


}

