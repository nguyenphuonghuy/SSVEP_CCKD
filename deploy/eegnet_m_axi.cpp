// H4 (mo-hinh-hoa Muc 5, G5): wrapper m_axi tay cho EEGNet, thay cho AXI-Stream
// + AXI DMA tu sinh boi hls4ml VivadoAccelerator (mac ket o TLAST tren cong ra
// sau 6 lan sua loi khac nhau cua thu vien). Theo dung mau g3_decode_rt.cpp
// cua khung G3.1 (ssvep_lite) da chay duoc tren chinh board nay: m_axi cho du
// lieu (offset=slave, 1 bundle gmem chung), s_axilite chi cho "return" (bundle
// control) de PS dieu khien ap_ctrl_hs (start/done) va cac thanh ghi dia chi
// m_axi tu dong duoc hls tu sinh vao cung bundle control.
//
// myproject() (trong myproject.cpp/h cua du an eegnet_1c_w10_resource_prj) da
// qua csim+csynth+vsynth sach (14.3% LUT) o ban build "Vitis" thuan truoc do;
// wrapper nay chi bao quanh no bang hls::stream noi bo, khong dong lai logic
// tinh toan.
#include "myproject.h"
#include "defines.h"

typedef ap_fixed<16, 9, AP_RND_CONV, AP_SAT, 0> mem_in_t;
typedef ap_fixed<16, 6, AP_RND_CONV, AP_SAT, 0> mem_out_t;

void eegnet_m_axi(mem_in_t *in_mem, mem_out_t *out_mem) {
#pragma HLS INTERFACE m_axi port = in_mem offset = slave bundle = gmem depth = 250
#pragma HLS INTERFACE m_axi port = out_mem offset = slave bundle = gmem depth = 40
#pragma HLS INTERFACE s_axilite port = in_mem bundle = control
#pragma HLS INTERFACE s_axilite port = out_mem bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    hls::stream<input_t> in_local("in_local");
    hls::stream<result_t> out_local("out_local");
#pragma HLS STREAM variable = in_local depth = 250
#pragma HLS STREAM variable = out_local depth = 1

READ_LOOP:
    for (int i = 0; i < 250; i++) {
#pragma HLS PIPELINE II = 1
        input_t tmp;
        tmp[0] = (input_t::value_type)in_mem[i];
        in_local.write(tmp);
    }

    myproject(in_local, out_local);

    result_t r = out_local.read();
WRITE_LOOP:
    for (int j = 0; j < 40; j++) {
#pragma HLS PIPELINE II = 1
        out_mem[j] = (mem_out_t)r[j];
    }
}
