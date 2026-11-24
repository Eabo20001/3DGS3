#include "fill_with_order.h"
#include "cuda_runtime.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include <cooperative_groups.h>
#include <cstdio>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>


namespace cg = cooperative_groups;

__global__ void CUDA_FillWithOrder(
    int P,
    const float *original_value,
    const int *to_fill_indices,
    const float *order,
    float *filled_value,
    float *filled_order,
    int *locks)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    float current_val = original_value[idx];
    int to_fill_idx = to_fill_indices[idx];
    float current_order = order[idx];
    while (atomicCAS(locks + to_fill_idx, 0, 1))
        continue; // lock

    if (filled_order[to_fill_idx] > current_order)
    {
        filled_order[to_fill_idx] = current_order;
        filled_value[to_fill_idx] = current_val;
    }

    // unlock
    atomicCAS(locks + to_fill_idx, 1, 0);
}

void FillWithOrder(
    int num_ori, int num_fill,
    const float *original_value,
    const int *to_fill_indices,
    const float *order,
    float *filled_value,
    float *filled_order)
{
    int * locks;
    cudaMalloc(&locks, sizeof(int) * num_fill);
    cudaMemset(locks, 0, sizeof(int)*num_fill);
    CUDA_FillWithOrder<<<(num_ori + 255) / 256, 256>>>(num_ori, original_value, to_fill_indices, order, filled_value, filled_order, locks);
    cudaFree(locks);
    // for (int i = 0; i<num_ori; ++i){
    //     int to_fill_idx = to_fill_indices[i];
    //     if (order[i] < filled_order[to_fill_idx]){
    //         filled_order[to_fill_idx] = order[i];
    //         filled_value[to_fill_idx] = original_value[i];
    //     }

    // }
}

template <typename scalar_t>
__global__ void CUDA_WarpImage(
    int C,
    int H,
    int W,
    const scalar_t *original_value,
    const int *dx,
    const int *dy,
    const scalar_t *order,
    scalar_t *filled_value,
    scalar_t *filled_order,
    int *locks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= H || j >= W) 
        return;
    int id = i * W + j;
    int to_i = i + dx[id];
    int to_j = j + dy[id];
    if(to_i<0 || to_i>=H || to_j<0 || to_j>=W)
        return;
        scalar_t current_order = order[id];
    int to_id = to_i * W + to_j;
    while (atomicCAS(locks + to_id, 0, 1))
        continue; // lock

    if (filled_order[to_id] > current_order)
    {
        filled_order[to_id] = current_order;
        for(int c=0; c<C; c++)
            filled_value[to_id + c*H*W] = original_value[id + c*H*W];   
    }

    // unlock
    atomicCAS(locks + to_id, 1, 0);
}

void WarpImage(
    int C,
    int H,
    int W,
    const torch::Tensor & original_value,   // values to fill
    const torch::Tensor & dx,    // respective index x of filled_value
    const torch::Tensor & dy,    // respective index y of filled_value
    const torch::Tensor & order,            // order (smaller)
    torch::Tensor & filled_value,           // output buffer (filled value)
    torch::Tensor & filled_order            // output buffer (filled order)
)
{
    int * locks;
    cudaMalloc(&locks, sizeof(int) * H * W);
    cudaMemset(locks, 0, sizeof(int)* H * W);
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((H + threadsPerBlock.x) / threadsPerBlock.x, (W + threadsPerBlock.y) / threadsPerBlock.y);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(original_value.scalar_type(), "WarpImage", ([&] {
        const scalar_t * original_value_ = original_value.data<scalar_t>();
        const int * dx_ = dx.data_ptr<int>();
        const int * dy_ = dy.data_ptr<int>();
        const scalar_t * order_ = order.data_ptr<scalar_t>();
        scalar_t * filled_value_ = filled_value.data_ptr<scalar_t>();
        scalar_t * filled_order_ = filled_order.data_ptr<scalar_t>();

        CUDA_WarpImage<scalar_t>
            <<<numBlocks, threadsPerBlock>>>(
                C, H, W, original_value_, dx_, dy_, order_, filled_value_, filled_order_, locks
            );
      }));
    // CUDA_WarpImage<<<numBlocks, threadsPerBlock>>>(C, H, W, original_value, dx, dy, order, filled_value, filled_order, locks);
    cudaFree(locks);
}


__global__ void G_CUDA_WarpImage_FWD(
    int C,
    int H,
    int W,
    const float *original_value,
    const int *dx,
    const int *dy,
    const float *order,
    float *filled_value,
    float *filled_order,
    int *locks,
    int *source_buffer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= H || j >= W) 
        return;
    int id = i * W + j;
    int to_i = i + dx[id];
    int to_j = j + dy[id];
    if(to_i<0 || to_i>=H || to_j<0 || to_j>=W)
        return;
    float current_order = order[id];
    int to_id = to_i * W + to_j;
    while (atomicCAS(locks + to_id, 0, 1))
        continue; // lock

    if (filled_order[to_id] > current_order)
    {
        filled_order[to_id] = current_order;
        source_buffer[to_id] = id;
        for(int c=0; c<C; c++)
            filled_value[to_id + c*H*W] = original_value[id + c*H*W];   
    }

    // unlock
    atomicCAS(locks + to_id, 1, 0);
}

__global__ void G_CUDA_WarpImage_BWD(
    int C,
    int H,
    int W,
    const float *grad_pix,
    const int *source_buffer,
    float *grad
    )
{
    // one pixel has only one source
    // no need for any lock
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    int j = blockIdx.y * blockDim.y + threadIdx.y; 
    if (i >= H || j >= W) 
        return;
    int id = i * W + j;
    if (source_buffer[id] < 0) return; // no source pix
    for (int c=0; c<C; ++c){
        grad[source_buffer[id] + c*H*W] = grad_pix[id + c*H*W];
    }

}

void GWarpImage_F(
    int C,
    int H,
    int W,
    const float * original_value,   // values to fill
    const int * dx,    // respective index x of filled_value
    const int * dy,    // respective index y of filled_value
    const float * order,            // order (smaller)
    float * filled_value,           // output buffer (filled value)
    float * filled_order,            // output buffer (filled order)
    int * source_buffer            // output buffer (filled order)
)
{
    int * locks;
    cudaMalloc(&locks, sizeof(int) * H * W);
    cudaMemset(locks, 0, sizeof(int)* H * W);
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((H + threadsPerBlock.x) / threadsPerBlock.x, (W + threadsPerBlock.y) / threadsPerBlock.y);
    G_CUDA_WarpImage_FWD<<<numBlocks, threadsPerBlock>>>(C, H, W, original_value, dx, dy, order, filled_value, filled_order, locks, source_buffer);
    cudaFree(locks);
}

void GWarpImage_B(
    int C,
    int H,
    int W,
    const float * grad_in,   // values to fill
    const int * source_buffer     ,       // order (smaller)
    float * output           // output buffer (filled value)
)
{
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((H + threadsPerBlock.x) / threadsPerBlock.x, (W + threadsPerBlock.y) / threadsPerBlock.y);
    G_CUDA_WarpImage_BWD<<<numBlocks, threadsPerBlock>>>(C, H, W, grad_in, source_buffer, output);
}