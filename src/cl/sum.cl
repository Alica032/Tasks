#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define WORK_GROUP_SIZE 128


__kernel void sum(__global const unsigned int* data_in, __global unsigned int* data_out, const unsigned int n){
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local unsigned int loc_mem[WORK_GROUP_SIZE];

    loc_mem[localId] = globalId < n? data_in[globalId]:0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(unsigned int nvalues = WORK_GROUP_SIZE/2; nvalues>0; nvalues/=2){
        if(localId < nvalues)
            loc_mem[localId] += loc_mem[localId+nvalues];
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if(localId==0)
        atomic_add(data_out, loc_mem[0]);
}


