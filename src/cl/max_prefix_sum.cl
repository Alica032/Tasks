#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 128

__kernel void max_prefix_sum(__global const int* data, unsigned int n, __global int* result) {
    const unsigned int globId = get_global_id(0);
    const unsigned int locId = get_local_id(0);
    const unsigned int workGroup = get_local_size(0);
    const unsigned int workGroupId = get_group_id(0);
    const unsigned count_group = (n + workGroup - 1) / workGroup;


    __local int loc_max[WORK_GROUP_SIZE];
    __local int loc_sum[WORK_GROUP_SIZE];
//    __local  int loc_idx[WORK_GROUP_SIZE];

    loc_max[locId] = globId < n? data[globId]: 0;
    loc_sum[locId] = globId < n? data[globId + n]: 0;
//    loc_idx[local_id] = globId;


    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int lvl = 1; lvl < WORK_GROUP_SIZE; lvl *= 2) {

        unsigned int id = 2*lvl*locId;

        if (id + lvl < WORK_GROUP_SIZE) {

            loc_max[id] = max(max(loc_max[id], loc_sum[id] + loc_max[id + lvl]), 0);
            loc_sum[id] += loc_sum[id + lvl];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(locId == 0) {
        result[workGroupId] = loc_max[0];
        result[workGroupId + count_group] = loc_sum[0];
    }
}


