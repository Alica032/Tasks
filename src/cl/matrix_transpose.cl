
#define TITLE_SIZE 16

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    __local float tile[TITLE_SIZE*TITLE_SIZE];
    int local_i = get_local_id(0);
    int local_j = gel_local_id(1);
    unsigned int mGroupSize = get_local_size(0);

    if(global_i < m && global_j < k)
    	tile[local_j*TITLE_SIZE + local_i] = a[global_j*k + global_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int at_i = get_group_id(1) * TITLE_SIZE + local_i;
    const int at_j = get_group_id(0) * TITLE_SIZE + local_j;

    if(at_i < m && at_j < k)
    	at[at_j*TITLE_SIZE + at_i] = tile[local_j*TITLE_SIZE + local_i];

}

