#pragma once

extern "C" int astr_keep_w1_prepare(void *handle);
extern "C" int astr_keep_w1_diffusion(void *handle,int first_y,int total_y,const int split[3],
    const double inverse[3],const double wall[2],double re,double pr,double c5,double tc,double tc1,
    const int send_width[2],const int recv_width[2]);

namespace astr {
struct State;
int exchange_keep_w1_ends(State &state,const int send_width[2],const int recv_width[2],const void *received[2]);
}
