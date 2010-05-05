/*******************************************************************************
 * Copyright (c) 2008-2009 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

/* $Revision: 10424 $ on $Date: 2010-02-17 14:34:49 -0800 (Wed, 17 Feb 2010) $ */

#ifndef __CL_EXT_H
#define __CL_EXT_H

#ifdef __cplusplus
extern "C" {
#endif

/* cl_khr_fp64 extension - no extension #define since it has no functions  */
#define CL_DEVICE_DOUBLE_FP_CONFIG                  0x1032


/* cl_khr_fp16 extension - no extension #define since it has no functions  */
#define CL_DEVICE_HALF_FP_CONFIG                    0x1033


/* cl_khr_icd extension                                                    */
#define cl_khr_icd 1

/* cl_platform_info                                                        */
#define CL_PLATFORM_ICD_SUFFIX_KHR                  0x0920
#define CL_PLATFORM_ICD_SUFFIX_NV                   0x0905

/* Additional Error Codes                                                  */
#define CL_PLATFORM_NOT_FOUND_KHR                   -1001

extern CL_API_ENTRY cl_int CL_API_CALL
clIcdGetPlatformIDsKHR(cl_uint          /* num_entries */,
                       cl_platform_id * /* platforms */,
                       cl_uint *        /* num_platforms */);


/******************************************************************************/
/* AMD Device attribute query extension */

#define cl_amd_device_attribute_query 1

#define CL_DEVICE_PROFILING_TIMER_OFFSET_AMD        0x403F

/******************************************************************************/
/* Device Fission Extension */

#define cl_ext_device_fission 1

/******************************************************************************/

typedef cl_uint cl_device_partition_property_ext;

/******************************************************************************/

/* Error Codes */
#define CL_INVALID_PROPERTY_EXT                     -1018
#define CL_DEVICE_PARTITION_FAILED_EXT              -1019
#define CL_INVALID_PARTITION_COUNT_EXT              -1020

/* cl_device_info */
#define CL_DEVICE_PARENT_DEVICE_EXT                 0x4030
#define CL_DEVICE_PARTITION_STYLE_EXT               0x4031

/* cl_device_partition_property_ext */
#define CL_DEVICE_PARTITION_EQUALLY_EXT             0x4032
#define CL_DEVICE_PARTITION_BY_COUNTS_EXT           0x4033
#define CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT  0x4034

/* cl_affinity_domain_ext */
#define CL_AFFINITY_DOMAIN_NUMA_EXT                 0x1
#define CL_AFFINITY_DOMAIN_L4_CACHE_EXT             0x2
#define CL_AFFINITY_DOMAIN_L3_CACHE_EXT             0x3
#define CL_AFFINITY_DOMAIN_L2_CACHE_EXT             0x4
#define CL_AFFINITY_DOMAIN_L1_CACHE_EXT             0x5
#define CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT     0x6

/* Device APIs */
typedef CL_API_ENTRY cl_int (CL_API_CALL * clCreateSubDevicesEXT_fn)(
    cl_device_id     /* in_device */,
    const cl_device_partition_property_ext * /* partition_properties */,
    cl_uint          /* num_entries */,
    cl_device_id *   /* out_devices */,
    cl_uint *        /* num_devices */);

typedef CL_API_ENTRY cl_int (CL_API_CALL * clRetainDeviceEXT_fn)(
    cl_device_id     /* device */) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL * clReleaseDeviceEXT_fn)(
    cl_device_id     /* device */) CL_API_SUFFIX__VERSION_1_0;

#ifdef __cplusplus
}
#endif

#endif /* __CL_EXT_H */
