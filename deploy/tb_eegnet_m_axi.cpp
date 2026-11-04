// Testbench toi thieu cho csim: chi kiem wrapper chay het, khong treo, dau ra
// khong NaN/vo nghia. Do chinh xac thuc se kiem bang cosim/chay tren board
// voi du lieu test thuc (xem deploy/h4_board_infer_maxi.py).
#include <cstdio>

#include "myproject.h"
#include "defines.h"

typedef ap_fixed<16, 9, AP_RND_CONV, AP_SAT, 0> mem_in_t;
typedef ap_fixed<16, 6, AP_RND_CONV, AP_SAT, 0> mem_out_t;

void eegnet_m_axi(mem_in_t *in_mem, mem_out_t *out_mem);

int main() {
    mem_in_t in_mem[250];
    mem_out_t out_mem[40];

    for (int i = 0; i < 250; i++) {
        double v = 0.3 * (i % 17 - 8);  // tin hieu gia, khong phai EEG thuc
        in_mem[i] = (mem_in_t)v;
    }

    eegnet_m_axi(in_mem, out_mem);

    printf("out_mem: ");
    for (int j = 0; j < 40; j++) {
        printf("%f ", out_mem[j].to_double());
    }
    printf("\n");
    return 0;
}
