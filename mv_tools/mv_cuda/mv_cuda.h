#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


void ForwardMv(
    size_t pW, size_t pH, float p_fx, float p_fy, // intrinsics
    const float *p_prev_w2c_matrix,
    const float *p_curr_w2c_matrix,
    const float *p_prev_depth,
    float *output);

void BackwardMv(
    size_t pW, size_t pH, float p_fx, float p_fy, // intrinsics
    const float *p_prev_w2c_matrix,
    const float *p_curr_w2c_matrix,
    const float *p_curr_depth,
    float *output);

void GetInvJacobi_Fwdmv(
    size_t pW, size_t pH, 
    const float* p_mv,
    float * output 
);