# H4 (mo-hinh-hoa Muc 5, G5): dung block design Vivado cho IP eegnet_m_axi
# (wrapper m_axi tay, thay AXI-Stream/DMA cua hls4ml). Theo dung mau
# build_overlay.tcl cua khung G3.1 (ssvep_lite), don gian hoa cho 1 IP duy nhat.
# Chay: vivado -mode batch -source build_overlay_maxi.tcl
set_param general.maxThreads 2 ;# host ubuntu-huy 14 GB RAM
set root [file normalize .]
create_project eegnet_ov $root/vivado_eegnet_maxi -part xczu7ev-ffvc1156-2-e -force
set bp [lindex [lsort [get_board_parts -quiet *zcu104*]] end]
puts "board_part = $bp"
set_property board_part $bp [current_project]
set_property ip_repo_paths $root/eegnet_maxi_prj/sol/impl/ip [current_project]
update_ip_catalog

create_bd_design eegnet
create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == zynq_ultra_ps_e}] ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100}] [get_bd_cells ps]

create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == eegnet_m_axi}] eegnet_0

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {/ps/pl_clk0} Clk_slave {/ps/pl_clk0} Clk_xbar {/ps/pl_clk0} \
   Master {/ps/M_AXI_HPM0_FPD} Slave /eegnet_0/s_axi_control ddr_seg {Auto} intc_ip {Auto} master_apm {0}] [get_bd_intf_pins eegnet_0/s_axi_control]

# Duong du lieu: noi tay qua smartconnect -> ps/S_AXI_HP0_FPD, dung mau da chay
# duoc cua g3_decode_rt (build_overlay.tcl), KHONG dung apply_bd_automation (rule
# axi4 bao "No valid slave interface" voi HP0 tren Vivado 2024.2 nay).
create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == smartconnect}] sc_mem
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_mem]
connect_bd_intf_net [get_bd_intf_pins eegnet_0/m_axi_gmem] [get_bd_intf_pins sc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_mem/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins sc_mem/aclk] [get_bd_pins ps/saxihp0_fpd_aclk]
set rst [lindex [get_bd_cells -filter {VLNV =~ *proc_sys_reset*}] 0]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins sc_mem/aresetn]

assign_bd_address
validate_bd_design
save_bd_design
set wrap [make_wrapper -files [get_files eegnet.bd] -top]
add_files -norecurse $wrap
set_property top eegnet_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
puts "IMPL_STATUS = [get_property STATUS [get_runs impl_1]]"
open_run impl_1
file mkdir $root/overlay_eegnet_maxi
report_timing_summary -file $root/overlay_eegnet_maxi/timing.rpt
report_utilization -file $root/overlay_eegnet_maxi/util.rpt
puts "WNS = [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
file copy -force $root/vivado_eegnet_maxi/eegnet_ov.runs/impl_1/eegnet_wrapper.bit $root/overlay_eegnet_maxi/eegnet_maxi.bit
file copy -force [glob $root/vivado_eegnet_maxi/eegnet_ov.gen/sources_1/bd/eegnet/hw_handoff/eegnet.hwh] $root/overlay_eegnet_maxi/eegnet_maxi.hwh
puts "OVERLAY_DONE"
