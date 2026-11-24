#include "mv_cuda.h"
#include "cuda_runtime.h"
#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__
#include <cooperative_groups.h>
#include<cstdio>

#define BSIZE 128

namespace cg = cooperative_groups;
__global__ void kForwardMv(
    size_t pW, size_t pH, float p_fx, float p_fy, // intrinsics
    const float *p_prev_w2c_matrixp,
    const float *p_curr_w2c_matrixp,
    const float *p_prev_depth,
    float *output)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx > pW * pH)
        return;

    const glm::mat4 p_prev_w2c_matrix_inv = glm::inverse(glm::make_mat4(p_prev_w2c_matrixp));
    const glm::mat4 p_curr_w2c_matrix = glm::make_mat4(p_curr_w2c_matrixp);

        size_t iy = idx / pW,
                    ix = idx % pW; // get the pixel position

    glm::vec3 dir((float(ix) - float(pW) / 2.f)*p_fx, (float(iy) - float(pH) / 2.f) *p_fy, 1.f);
    glm::vec3 c_coord = p_prev_depth[idx] * dir;
    glm::vec4 homo_c_coord(c_coord, 1.f);
    homo_c_coord = p_curr_w2c_matrix * p_prev_w2c_matrix_inv * homo_c_coord;
    homo_c_coord /= homo_c_coord.w;

    // write the coord output
#pragma unroll 3
    for (int i = 0; i < 3; ++i)
        output[5 * idx + i] = homo_c_coord[i];
    // convert from c_coord to rast_coord
    float tx = homo_c_coord.x / homo_c_coord.z / p_fx + float(pW) / 2.f;
    float ty = homo_c_coord.y / homo_c_coord.z / p_fy + float(pH) / 2.f;
    output[5 * idx + 3] = tx - ix;
    output[5*idx + 4] = ty - iy;
}

__global__ void kBackwardMv(
    size_t pW, size_t pH, float p_fx, float p_fy, // intrinsics
    const float *p_prev_w2c_matrixp,
    const float *p_curr_w2c_matrixp,
    const float *p_curr_depth,
    float *output)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx > pW * pH)
        return;

    const glm::mat4 p_prev_w2c_matrix = glm::make_mat4(p_prev_w2c_matrixp);
    const glm::mat4 p_curr_w2c_matrix_inv = glm::inverse(glm::make_mat4(p_curr_w2c_matrixp));
    size_t iy = idx / pW, ix = idx % pW; // get the pixel position

    glm::vec3 cam_coord((float(ix) - float(pW) / 2.f) * p_fx, (float(iy) - float(pH) / 2.f) * p_fy, 1.f);
    glm::vec3 dir((float(ix) - float(pW) / 2.f) * p_fx, (float(iy) - float(pH) / 2.f) * p_fy, 1.f);
    glm::vec3 c_coord = p_curr_depth[idx] * dir;
    glm::vec4 homo_c_coord(c_coord, 1.f);
    homo_c_coord = p_prev_w2c_matrix * p_curr_w2c_matrix_inv * homo_c_coord;
    homo_c_coord /= homo_c_coord.w;

    // write the coord output
#pragma unroll 3
    for (int i = 0; i < 3; ++i)
        output[5 * idx + i] = homo_c_coord[i];
    // convert from c_coord to rast_coord
    float tx = homo_c_coord.x / homo_c_coord.z / p_fx + float(pW) / 2.f;
    float ty = homo_c_coord.y / homo_c_coord.z / p_fy + float(pH) / 2.f;
    output[5 * idx+3] = tx - ix;
    output[5 * idx+4] = ty - iy;
}

void ForwardMv(
    size_t pW, size_t pH, float p_fx, float p_fy, // intrinsics
    const float *p_prev_w2c_matrixp,
    const float *p_curr_w2c_matrixp,
    const float *p_prev_depth,
    float *output)
{
    size_t num = pW * pH;
    kForwardMv<<<(num + BSIZE-1) / BSIZE, BSIZE>>>(pW, pH, p_fx, p_fy, p_prev_w2c_matrixp, p_curr_w2c_matrixp, p_prev_depth, output);
}

void BackwardMv(
    size_t pW, size_t pH, float p_fx, float p_fy, // intrinsics
    const float *p_prev_w2c_matrixp,
    const float *p_curr_w2c_matrixp,
    const float *p_curr_depth,
    float *output){
    size_t num = pW * pH;
    kBackwardMv<<<(num + BSIZE-1) / BSIZE, BSIZE>>>(pW, pH, p_fx, p_fy, p_prev_w2c_matrixp, p_curr_w2c_matrixp, p_curr_depth, output);
    }

__global__ void kGetInvJacobi_Fwdmv(
    size_t pW, size_t pH,
    const float *p_mv,
    float *poutput){
    auto idx = cg::this_grid().thread_rank();
    if (idx > pW * pH)
        return;
    size_t iy = idx / pW,
           ix = idx % pW; // get the pixel position

    float* offset = poutput + 4 * idx;
    
    // get diff
    float2 du, dv;
    if (ix < pW-1)
    {
        du.x = p_mv[2 * ((iy) * pW + ix + 1)] - p_mv[2 * idx] + 1;
        dv.x = p_mv[2 * ((iy) * pW + ix + 1) + 1] - p_mv[2 * idx + 1] + 0;
    }
    else
    {
        du.x = 1;
        dv.x = 0;
    }

    if (iy < pH - 1)
    {
        du.y = p_mv[2 * ((iy + 1) * pW + ix)] - p_mv[2 * idx] + 0;
        dv.y = p_mv[2 * ((iy + 1) * pW + ix)+1] - p_mv[2 * idx+1] + 1;
    }
    else
    {
        du.y = 0;
        dv.y = 1;
    }
    auto jac = glm::mat2x2(du.x, dv.x, du.y,dv.y);
    float det = glm::determinant(jac);
    if (fabs(det) < 1e-8)
        jac = glm::mat2x2(1.f, 0.f, 0.f, 1.f);

    auto invjac = glm::inverse(jac);
#pragma unroll 4
    for (int i = 0; i<4;++i){
        offset[i] = invjac[i%2][i/2];
    }
}

    void GetInvJacobi_Fwdmv(
        size_t pW, size_t pH,
        const float *p_mv,
        float *output)
{
    size_t num = pW * pH;
    kGetInvJacobi_Fwdmv<<<(num + BSIZE-1) / BSIZE, BSIZE>>>(pW, pH, p_mv, output);
}
