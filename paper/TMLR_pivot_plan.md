# Kế hoạch xoay trục bản thảo TMLR (tách khỏi bài ICERA 2026)

Ngày soạn: 2026-10-20. Trạng thái: tác giả đã duyệt tiêu đề và RQ1–RQ3, chọn phạm vi chạy đầy đủ.
Bài ICERA 2026 đã được chấp nhận. Kế hoạch chạy ở mục 5 chờ xác nhận trước khi submit. Chưa sửa `main.tex`.

## 0. Nguyên tắc tách bài

- Bài ICERA 2026 (#271) đã công bố đóng góp "đề xuất CCKD, hiệu quả trên FB-tCNN". Bài TMLR
  không nhận lại đóng góp này.
- Câu hỏi chính của bài TMLR: khi nào và vì sao chưng cất quan hệ xuyên cấu hình điện cực
  (cross-montage relational KD) có ích.
- Kết quả FB-tCNN chỉ xuất hiện như một dòng tham chiếu trong bảng tổng hợp, ghi "reported in [ICERA]".
  TMLR chạy double-blind nên trích dẫn bài ICERA ở ngôi thứ ba.
- Phần phương pháp chỉ nhắc lại công thức Gram (Eq. 1–3 của ICERA) ở dạng rút gọn kèm trích dẫn.
  Không dùng lại Algorithm 1, Fig. 1, Fig. 2.
- Không dùng lại câu chữ của ICERA. Keywords cũng viết lại.

## 1. Tiêu đề đề xuất

*When Does Relational Distillation Transfer Across Electrode Montages? A Multi-Backbone Study
for Low-Electrode SSVEP Recognition*

## 2. Abstract (bản nháp tiếng Anh)

Ô `[TBD: ...]` là số liệu phụ thuộc thí nghiệm chưa chạy. Số liệu còn lại lấy từ main.pdf hiện tại
(Bảng 11, 21, 24 và văn bản mục 5.6).

> Knowledge distillation (KD) from a multi-electrode teacher can recover part of the accuracy lost
> when an SSVEP decoder uses fewer electrodes. Prior work reported such gains for cross-channel
> relational KD (CCKD) on a single backbone, FB-tCNN [ICERA]. It is unclear whether these gains
> hold for other architectures and what predicts them. We evaluate CCKD on five public backbones
> (FB-tCNN, DeepConvNet, ShallowConvNet, EEGNet, and Compact-CNN), two 40-target datasets
> (Benchmark and BETA), and two student montages with three and one electrodes. Every setting uses
> five independent subject splits with eight seeds per split. With a fixed distillation weight,
> mean gains ranged from −1.30 to +1.19 percentage points (pp). Eight of the 20
> backbone-dataset-montage settings improved significantly, four degraded significantly, and eight
> showed no detectable effect (one-sample t-test over splits, p < 0.05). [TBD: result after
> per-backbone, validation-selected weights.] The gain tracked how much CCKD increased the linear
> CKA between teacher and student features (Spearman ρ = 0.63, p = 0.003, n = 20). The CKA
> before distillation did not predict the gain (ρ = 0.06, p = 0.80). [TBD: result of the
> distillation-layer intervention.] Evaluation on a single subject split misjudged the sign or
> significance of the effect in [TBD: k] of 20 settings. Run-to-run nondeterminism reached
> 0.51 pp on the gain, the same order as many reported effects. These results show that
> cross-montage relational KD is backbone-dependent. They also argue for multi-split evaluation
> when SSVEP KD gains are below 1 to 2 pp.

Ghi chú kiểm tra số liệu:
- "8 tăng, 4 giảm, 8 không phát hiện": đếm từ Bảng 21 với p < 0.05 chưa hiệu chỉnh.
  Tăng: FB-tCNN 4, DeepConvNet 3, EEGNet 1 (BM-3C). Giảm: ShallowConvNet 2 (BM), EEGNet 2 (BETA).
  Khi viết bài sẽ bổ sung con số sau hiệu chỉnh Holm.
- ρ = 0.627 và 0.060 trong main.pdf, làm tròn thành 0.63 và 0.06.
- 0.51 pp là SD liên tiến trình lớn nhất của ∆ trong Bảng 11 (Benchmark, 1C).
- Con số k cần đếm lại khi có bảng so sánh một split với nhiều split (bảng N6).

## 3. Outline mới

1. **Introduction**
   - Bài toán giảm điện cực SSVEP và KD từ teacher nhiều kênh.
   - Kết quả đã biết trên FB-tCNN [ICERA] và ba câu hỏi còn mở.
   - Ba câu hỏi nghiên cứu (RQ):
     - RQ1: Mức cải thiện có chuyển được sang backbone khác không?
     - RQ2: Đại lượng nào dự đoán mức cải thiện?
     - RQ3: Đánh giá trên một split có đáng tin không?
   - Đóng góp: (i) đánh giá có kiểm soát trên 5 backbone × 2 dataset × 2 montage × 5 split × 8 seed;
     (ii) chỉ báo ∆CKA kèm thí nghiệm can thiệp; (iii) phân tích độ tin cậy của đánh giá.
2. **Related Work**
   - Backbone học sâu cho SSVEP.
   - KD cho EEG và KD khi teacher và student có đầu vào khác nhau (cross-input KD, MESD).
   - Nghiên cứu về điều kiện KD có hiệu quả.
   - Đo tương đồng biểu diễn (CKA).
   - Độ tin cậy của đánh giá cross-subject trong EEG.
   - Mọi trích dẫn mới phải kiểm DOI trước khi đưa vào.
3. **Setup**
   - 3.1 CCKD tóm tắt kèm trích dẫn, một phương trình.
   - 3.2 Năm backbone và bảng phân loại kiến trúc (bảng N1).
   - 3.3 Dataset và montage.
   - 3.4 Giao thức: 5 split × 8 seed; chọn λ theo validation riêng cho từng backbone; kiểm định
     (t một mẫu, Holm, TOST).
4. **RQ1: Hiệu ứng theo backbone**: bảng tổng hợp (Bảng 21 hiện tại cộng cột pHolm) và λ theo
   từng backbone (bảng N2).
5. **RQ1b: Độ nhạy montage**: các tập điện cực thay thế (bảng N3).
6. **RQ2: Đại lượng dự đoán mức cải thiện**: ∆CKA nhiều seed và hai lớp (bảng N4), thí nghiệm can
   thiệp theo lớp distill (bảng N5).
7. **RQ3: Độ tin cậy của đánh giá**: một split so với nhiều split (hình N6), nondeterminism
   (Bảng 11), chọn λ trên một split (Bảng 10, 12).
8. **Discussion và Limitations**
9. **Conclusion**
- **Phụ lục**: bảng theo từng split, chi tiết siêu tham số, mã nguồn.

## 4. Bảng ánh xạ 26 bảng hiện có

Quy ước: **Giữ** = vào thân bài; **Giữ-PL** = đưa vào phụ lục; **Cắt** = bỏ (trùng ICERA hoặc lệch
câu hỏi mới); **Thay** = bỏ và thay bằng thí nghiệm chạy thêm.

| Bảng | Nội dung | Quyết định | Lý do |
|---|---|---|---|
| 1 | Tóm tắt thiết lập | Giữ (viết lại) | Thêm 5 backbone, montage, giao thức chọn λ. Tránh trùng Table 1 ICERA. |
| 2 | FB-tCNN, split 2026 | Cắt | Trùng Table 2 ICERA (trái). |
| 3 | FB-tCNN, 5 split | Cắt | Trùng Table 2 ICERA (phải). Chỉ giữ một dòng trong bảng tổng hợp, ghi trích dẫn. |
| 4 | Ablation chiến lược KD, Benchmark, 1 split | Cắt | Trùng nội dung Table 3 ICERA. |
| 5 | Logit KD theo τ, 5 split | Cắt | So sánh hàm mục tiêu trên FB-tCNN, ngoài câu hỏi mới. |
| 6 | So sánh KD, mỗi hàng một baseline riêng | Cắt | Baseline không đồng nhất, đã có bảng thay thế. |
| 7 | CCKD so với cơ chế KD khác, 1 split | Cắt | Trùng Table 3 ICERA. |
| 8 | FitNets, 5 split | Cắt | Trùng Table 3 ICERA. Nếu làm mục tùy chọn T-opt thì thay bằng bảng nhiều backbone. |
| 9 | Attention Transfer, 5 split | Cắt | Như Bảng 8. |
| 10 | Độ nhạy λ, FB-tCNN, split 2026 | Giữ-PL | Dữ liệu cho RQ3: λ chọn theo validation trên một split không ổn định. ICERA chưa có. |
| 11 | Nondeterminism | Giữ | RQ3. ICERA chỉ nhắc định tính. |
| 12 | λ = 1 so với λ = 100, 5 split | Giữ | RQ3: lựa chọn trên một split không tồn tại qua nhiều split. |
| 13 | Độ dài cửa sổ, 1 split | Cắt | Trùng hướng Table 4 ICERA. |
| 14 | Độ dài cửa sổ, 5 split | Cắt | Trùng Table 4 ICERA. |
| 15 | EEGNet, 1 split, 3 seed | Cắt | Thay bằng cột split 2026 của Bảng 16 (8 seed, nhất quán). |
| 16 | EEGNet, 5 split | Giữ | Gộp vào bảng tổng hợp RQ1; bảng theo split để phụ lục. |
| 17 | Quét λ trên EEGNet, 3 seed | Thay | Thay bằng N2 (λ theo validation cho cả 5 backbone, 8 seed, 5 split). |
| 18 | Chiến lược thay thế trên EEGNet | Cắt | Lệch câu hỏi; λ_sp = 5 không theo giao thức chung. |
| 19 | DeepConvNet, 1 split | Cắt | Thay bằng cột split 2026 của Bảng 23. |
| 20 | ShallowConvNet và Compact-CNN, 1 split | Cắt | Như Bảng 19. |
| 21 | Tổng hợp 5 backbone, 5 split | Giữ | Bảng trung tâm của RQ1. Thêm cột pHolm. Dòng FB-tCNN ghi "[ICERA]". |
| 22 | Số kết quả sống sót sau Holm | Giữ (rút gọn) | Bỏ các dòng một split. Có thể gộp thành cột của Bảng 21. |
| 23 | DeepConvNet, ShallowConvNet, Compact-CNN theo split | Giữ-PL | Số liệu theo từng split cho Bảng 21. |
| 24 | CKA, 1 seed, lớp cuối | Thay | Giữ phân tích, chạy lại nhiều seed và hai lớp (N4). |
| 25 | Độ trễ Raspberry Pi 4 | Cắt | Trùng mục 5.4 ICERA. Chỉ nhắc một câu kèm trích dẫn. |
| 26 | ITR có cộng độ trễ | Cắt | Thuộc nội dung triển khai đã công bố. |

Tổng: Giữ 6 (1, 11, 12, 16, 21, 22); Giữ-PL 2 (10, 23); Thay 2 (17, 24);
Cắt 16.

## 5. Thí nghiệm chạy thêm (phạm vi đầy đủ)

### 5.1 Lỗi thiết kế phát hiện khi đọc code (phải xử lý trước)

`models/cckd_backbones.py` trả về hai lớp đặc trưng cho FB-tCNN (`[feat_mid, feat]`) nhưng chỉ
một lớp cho EEGNet, DeepConvNet, ShallowConvNet và Compact-CNN (`[feat]`). `CCKDLoss` lấy trung
bình loss trên các lớp được trả về. Như vậy Bảng 21 hiện so FB-tCNN distill hai lớp với bốn backbone
distill một lớp. Khác biệt giữa các backbone đang lẫn với khác biệt số lớp distill.

Hướng xử lý:
- Giao thức chính: mọi backbone distill **chỉ lớp cuối** (lớp penultimate). Bốn backbone đã có sẵn
  dữ liệu theo giao thức này. FB-tCNN cần chạy lại ở chế độ chỉ lớp cuối.
- Giao thức can thiệp (N5): mọi backbone distill **lớp giữa**, **lớp cuối** và **cả hai**. Lớp giữa
  được định nghĩa thống nhất là đầu ra ngay sau khối gộp kênh không gian (spatial collapse). Cách
  định nghĩa này cũng là biến giải thích cho RQ2.
- Kết quả FB-tCNN hai lớp vẫn là kết quả đã công bố ở ICERA, chỉ dùng để đối chiếu.

### 5.2 Sửa code (không cần GPU)

| Mã | Thay đổi | File |
|---|---|---|
| C1 | Thêm `return_feat` trả về lớp giữa cho EEGNet, DeepConvNet, ShallowConvNet, Compact-CNN | `models/cckd_backbones.py` |
| C2 | Thêm tham số `--kd_layers {last,mid,both}` | `training/train_cckd.py`, `training/cckd_pipeline.py` |
| C3 | Thêm montage thay thế vào `CH_POS` và cho phép chọn qua `--students` | `training/cckd_pipeline.py` |
| C4 | Ghi CKA tuyến tính (teacher–student, cả hai lớp, trên tập test) cho mọi lần chạy | `training/cckd_pipeline.py` |
| C5 | Đặt tên file kết quả theo `{dataset}_{model}_{setting}_seed{N}_{YYYYMMDD}.csv`, lưu config JSON cạnh kết quả | `training/train_cckd.py` |
| C6 | Kiểm thử: smoke test 1 epoch cho mỗi backbone × mỗi `--kd_layers` qua PBS | `tests/` |

C4 giúp N4 gần như không tốn thêm GPU, vì CKA được tính ngay trong N2, N3, N5 trên đủ 5 split × 8 seed.

Montage đề xuất cho N3. Thứ tự kênh teacher: Pz, PO5, PO3, POz, PO4, PO6, O1, Oz, O2.
- 3C gốc {O1, Oz, O2}. 3C thay thế: {PO3, POz, PO4} (dịch lên vùng đỉnh–chẩm) và {POz, O1, O2}.
- 1C gốc {Oz}. 1C thay thế: {POz} và {O1}.

### 5.3 Danh sách thí nghiệm

Một job = 1 backbone × 1 dataset × 1 split × 1 cấu hình × 8 seed (2026–2033). Thời gian mỗi job
lấy từ log cũ trên ai01: FB-tCNN khoảng 1,4–1,9 giờ, bốn backbone còn lại khoảng 0,5–0,8 giờ.

| Mã | Nội dung | Số job | GPU-giờ | Thứ tự |
|---|---|---|---|---|
| N0 | Smoke test sau khi sửa code (C6) | 10 | ~2 | 1 |
| N2a | FB-tCNN chỉ lớp cuối, λ = 100, 2 dataset × 5 split | 10 | ~17 | 2 |
| N2b | Quét λ ∈ {1, 10, 100, 1000}, chỉ lớp cuối, 5 backbone × 2 dataset × 5 split; chọn λ theo validation trên từng split (không dùng test) | ~150 mới (λ = 100 của 4 backbone đã có) | ~150 | 3 |
| N5 | Distill lớp giữa và cả hai lớp, 5 backbone × 2 dataset × 5 split, λ theo N2b | 100 | ~90 | 4 |
| N3 | Montage thay thế (4 student/job), 5 backbone × 2 dataset × 5 split, λ theo N2b | 50 | ~80 | 4 |
| T-opt | FitNets và AT trên 4 backbone còn lại, chỉ lớp cuối, 2 dataset × 5 split | 40 | ~40 | 5 |
| N4 | CKA: lấy từ C4, không cần job riêng | 0 | 0 | – |
| N6 | So sánh một split với nhiều split: tính lại từ số liệu có sẵn | 0 | 0 | – |
| **Tổng** | | **~360** | **~380** | |

Thời gian thực tế: nếu chạy song song 6–8 job trên `sharedgpu`/`sharedgpu2`, mất khoảng 3–5 ngày.
Có điều kiện phụ thuộc: N5 và N3 dùng λ chọn từ N2b, nên phải chờ N2b xong.

Output: `results/module3/<mã thí nghiệm>/<dataset>_<backbone>_<cấu hình>_s<split>/`. Log:
`logs/m3_<mã>_<...>_${PBS_JOBID}.log`. Mỗi thư mục có file config JSON. Không ghi đè thư mục
`results/module2/`.

PBS theo chuẩn máy: queue `sharedgpu`, `select=1:ncpus=4:mem=32gb:gpu_slots=1`,
`CUDA_MPS_PIPE_DIRECTORY=/dev/null` như các job tier6 đã chạy ổn.

## 6. Phần cứng: Raspberry Pi 4 và ZCU104 (RQ4)

Cập nhật 2026-10-20: Vivado/Vitis chạy được trên `ubuntu-huy` (cùng mạng Tailscale). ZCU104 cũng gắn vào
máy này. Như vậy rào cản công cụ cho hướng PL không còn.

> **RQ4.** Mức cải thiện của CCKD có còn khi mô hình chạy ở độ chính xác thấp (INT8 trên CPU,
> fixed-point trên FPGA), và lựa chọn backbone thay đổi thế nào khi xét cả độ chính xác, độ trễ và
> tài nguyên?

### 6.1 Rào cản kỹ thuật còn lại: kiến trúc mô hình

Kernel thời gian trong code hiện tại (`models/cckd_backbones.py`):

| Backbone | Kernel thời gian lớn nhất | Activation | Ghi chú |
|---|---|---|---|
| FB-tCNN | 250 (conv2, bằng cả cửa sổ), 46 (conv4) | ELU | |
| EEGNet | 64 | ELU | depthwise, separable |
| Compact-CNN | 64 | ELU | như EEGNet, F1 = 96 |
| ShallowConvNet | 25, pool 35 | square, log | |
| DeepConvNet | 10 | ELU | |

Theo hiểu biết của tôi, DPU của Vitis AI (DPUCZDX8G) chỉ nhận kernel tối đa khoảng 16 và không chạy ELU,
square, log. Hai giới hạn này cần đối chiếu lại với tài liệu của đúng phiên bản Vitis AI trên `ubuntu-huy`.
Nếu đúng, chỉ phần nhỏ của đồ thị chạy được trên DPU, phần còn lại rơi về CPU. Kết quả đo khi đó phản ánh
CPU nhiều hơn FPGA. Muốn chạy được trên DPU phải đổi kiến trúc, tức là đổi đối tượng nghiên cứu của RQ1.

Hướng hợp hơn là hls4ml hoặc HLS tự viết. Hai hướng này nhận kernel tùy ý và ELU, nhưng tốn tài nguyên với
kernel 250 của FB-tCNN. Cần ước lượng tài nguyên trước khi cam kết.

### 6.2 Phạm vi đề xuất cho bài TMLR

| Mã | Nội dung | Nơi chạy | Vai trò |
|---|---|---|---|
| H1 | INT8 PTQ (ONNX Runtime/TFLite) cho baseline và CCKD, 5 backbone × 2 dataset × 2 montage, test trên 5 split; so ∆ trước và sau lượng tử hóa | HPC (CPU, PBS) | Bắt buộc |
| H2 | Mô phỏng fixed-point bit-accurate bằng hls4ml, quét độ rộng bit (ví dụ 16, 12, 10, 8 bit); tìm độ rộng bit tại đó ∆ biến mất | HPC (CPU, PBS); không cần Vivado | Bắt buộc |
| H3 | Độ trễ và bộ nhớ FP32/INT8, batch 1, cửa sổ 1,0 s, 5 backbone | Pi 4, ZCU104 (PS Cortex-A53) | Bắt buộc |
| H4 | Một backbone tổng hợp đầy đủ trên PL (backbone nhỏ nhất khả thi), đo độ trễ, tài nguyên (LUT/DSP/BRAM), công suất và xác minh độ chính xác trên board | `ubuntu-huy` (Vivado/Vitis HLS), ZCU104 | Tùy chọn, làm bằng chứng khả thi |

H2 là phần làm RQ4 khác hẳn ICERA. Nó trả lời câu hỏi có ý nghĩa với cả cộng đồng ML: mức cải thiện nhỏ do
KD có tồn tại ở độ chính xác số học thấp không.

Không đưa vào bài TMLR: bộ tăng tốc PL tối ưu cho cả 5 backbone. Phần đó là một bài phần cứng riêng. Có
thể dùng chung luồng HLS/Vivado cho ZCU104 đang làm ở `ssvep_lite` (gv_dnton).

### 6.3 Hạ tầng đã kiểm (2026-10-20)

- Kết nối: gv_bvnam → `ubuntu-huy` → ZCU104 đã thông bằng khóa `~/.ssh/id_ed25519_nhung`.
- `ubuntu-huy` (hostname `Huythuong`): Ubuntu 22.04.5, x86_64, 16 core, 14 GB RAM, 195 GB trống.
  Có Vivado, Vitis, Vitis HLS **2024.2** ở `/tools/Xilinx` (chưa nạp vào PATH mặc định).
  Không có Docker image Vitis AI. Chưa cài hls4ml.
- Board ZCU104: PYNQ **3.0.1** (venv `/usr/local/share/pynq-venv`), RAM 2 GB, 19 GB trống.
  Chưa cài `pynq_dpu`/VART. Có cảm biến công suất `ina226` và `irps5401` qua hwmon, nên đo công suất được.
- Đã có sẵn luồng Vitis HLS → Vivado → overlay PYNQ cho ZCU104 của `ssvep_lite` (G3.1):
  `~/ssvep_lite_hls_g31` trên `ubuntu-huy` (có báo cáo timing, utilization, power cho T = 100 và T = 250)
  và `~/ssvep_lite_pkg_g31/pl_rt` trên board. H4 nên dùng lại khung này thay vì đi đường DPU.
  Đây là sản phẩm của NCS Đào Ngọc Tôn: cần thống nhất với Tôn về việc dùng lại và ghi công.

Hệ quả: đường DPU cần cài thêm Vitis AI và DPU-PYNQ, và vẫn vướng giới hạn kernel/activation ở mục 6.1.
Đường HLS đã chạy được trên đúng board này, nên H4 chọn đường HLS.

Việc còn lại: bật lại Pi 4 trên Tailscale trước khi đo H3.

## 7. Trạng thái quyết định

- [x] Tiêu đề và RQ1–RQ3: đã duyệt (2026-10-20).
- [x] Phạm vi chạy: đầy đủ.
- [x] Bài ICERA 2026: đã được chấp nhận. Bài TMLR phải trích dẫn ở ngôi thứ ba và khai báo trong
  cover letter. Cần kiểm tra điều khoản bản quyền của nhà xuất bản trước khi dùng lại bất kỳ hình nào.
- [ ] Đồng ý xử lý lỗi thiết kế 5.1 (giao thức chính "chỉ lớp cuối").
- [ ] Đồng ý phạm vi RQ4 ở mục 6.2 (H1–H3 bắt buộc, H4 tùy chọn).
- [x] Kết nối SSH tới `ubuntu-huy` và ZCU104, đã kiểm phiên bản công cụ (mục 6.3).
- [ ] Thống nhất với NCS Đào Ngọc Tôn về việc dùng lại luồng HLS của `ssvep_lite` cho H4.
- [ ] Xác nhận kế hoạch chạy 5.3 để bắt đầu sửa code và submit N0.
