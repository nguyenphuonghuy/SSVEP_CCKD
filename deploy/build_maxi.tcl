# H4 (mo-hinh-hoa Muc 5, G5): tong hop wrapper m_axi tay cho EEGNet.
# Vitis HLS 2024.2. Chay: vitis_hls -f build_maxi.tcl
set INC "-I firmware -I firmware/nnet_utils -std=c++14"
open_project -reset eegnet_maxi_prj
set_top eegnet_m_axi
add_files firmware/myproject.cpp -cflags $INC
add_files src/eegnet_m_axi.cpp -cflags $INC
add_files -tb src/tb_eegnet_m_axi.cpp -cflags $INC
open_solution -reset sol -flow_target vivado
set_part xczu7ev-ffvc1156-2-e
create_clock -period 10 -name default
csynth_design
export_design -format ip_catalog
exit
