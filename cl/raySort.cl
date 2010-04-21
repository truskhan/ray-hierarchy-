#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void raySort (
    __global float* dir, __global float* o, __global int* indices, int count, int i) {
    // find position in global arrays
    int iGID = get_global_id(0);

    if ( i == 0 ){
      while ( iGID < count) {
          indices[iGID] = iGID;
          iGID += get_global_size(0)+1; //plus total number of threads
      }

      iGID = get_global_id(0);
    }

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (3*iGID+2+i > count ) return;

    float4 dira, dirb, dirc, oa,ob,oc;
    float4 tempb, tempc;
    float distb, distc;
    int temp;
    unsigned int a1,a2,a3,b1,b2,b3,c1,c2,c3;
    if ( i%4 == 0) {
        a1 = 3;
        a2 = 4;
        a3 = 5;
        b1 = 0;
        b2 = 1;
        b3 = 2;
        c1 = 6;
        c2 = 7;
        c3 = 8;
    }
    if ( i%4 == 1) {
        a1 = 6;
        a2 = 7;
        a3 = 8;
        b1 = 3;
        b2 = 4;
        b3 = 5;
        c1 = 9;
        c2 = 10;
        c3 = 11;
    }
    if ( i%4 == 2) {
        a1 = 9;
        a2 = 10;
        a3 = 11;
        b1 = 6;
        b2 = 7;
        b3 = 8;
        c1 = 12;
        c2 = 13;
        c3 = 14;
    }
    if ( i%4 == 3) {
        a1 = 12;
        a2 = 13;
        a3 = 14;
        b1 = 9;
        b2 = 10;
        b3 = 11;
        c1 = 15;
        c2 = 16;
        c3 = 17;
    }

    dira = (float4)(dir[9*iGID+a1], dir[9*iGID+a2], dir[9*iGID+a3],0);
    dirb = (float4)(dir[9*iGID+b1], dir[9*iGID+b2], dir[9*iGID+b3],0);
    dirc = (float4)(dir[9*iGID+c1], dir[9*iGID+c2], dir[9*iGID+c3],0);
    oa = (float4)(o[9*iGID+a1], o[9*iGID+a2], o[9*iGID+a3],0);
    ob = (float4)(o[9*iGID+b1], o[9*iGID+b2], o[9*iGID+b3],0);
    oc = (float4)(o[9*iGID+c1], o[9*iGID+c2], o[9*iGID+c3],0);
    //tempb = fdim(dira,dirb)+2*fdim(oa,ob);
    tempb = fdim(oa,ob);
    distb = tempb[0] + tempb[1] + tempb[2];
    //tempc = fdim(dira,dirc) + 2*fdim(oa,oc);
    tempc = fdim(oa,oc);
    distc = tempc[0] + tempc[1] + tempc[2];
    temp = indices[3*iGID+i];
    if ( distb > distc) {
        //swap
        indices[3*iGID+i] = indices[3*iGID+2+i];
        indices[3*iGID+2+i] = temp;
        dir[9*iGID+a1] = dirc[0];
        dir[9*iGID+a2] = dirc[1];
        dir[9*iGID+a3] = dirc[2];
        dir[9*iGID+c1] = dira[0];
        dir[9*iGID+c2] = dira[1];
        dir[9*iGID+c3] = dira[2];
        o[9*iGID+a1] = oc[0];
        o[9*iGID+a2] = oc[1];
        o[9*iGID+a3] = oc[2];
        o[9*iGID+c1] = oa[0];
        o[9*iGID+c2] = oa[1];
        o[9*iGID+c3] = oa[2];
    } else {
        //swap
        indices[3*iGID+i] = indices[3*iGID+1+i];
        indices[3*iGID+1+i] = temp;
        dir[9*iGID+a1] = dirb[0];
        dir[9*iGID+a2] = dirb[1];
        dir[9*iGID+a3] = dirb[2];
        dir[9*iGID+b1] = dira[0];
        dir[9*iGID+b2] = dira[1];
        dir[9*iGID+b3] = dira[2];
        o[9*iGID+a1] = ob[0];
        o[9*iGID+a2] = ob[1];
        o[9*iGID+a3] = ob[2];
        o[9*iGID+b1] = oa[0];
        o[9*iGID+b2] = oa[1];
        o[9*iGID+b3] = oa[2];
    }


}


__kernel void rayResort ( const __global int* indices,
                          __global float* dir, __global float* o, int count) {
    // find position in global arrays
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= count) return;

    int i = indices[iGID];
    float4 dira, oa;
    dira = (float4)(dir[iGID*3],dir[iGID*3+1],dir[iGID*3+2],0);
    oa = (float4)(o[iGID*3],o[iGID*3+1],o[iGID*3+2],0);

    barrier(CLK_GLOBAL_MEM_FENCE);

    dir[3*i] = dira[0];
    dir[3*i+1] = dira[1];
    dir[3*i+2] = dira[2];
    o[3*i] = oa[0];
    o[3*i+1] = oa[1];
    o[3*i+2] = oa[2];
}
