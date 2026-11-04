# Kế hoạch xoay trục bản thảo IJRCS (mở rộng từ bài ICERA 2026)

Ngày soạn: 2026-11-01. Phương án thay thế cho `TMLR_pivot_plan.md` — chưa chọn nơi nộp cuối cùng,
tài liệu này dựng để so sánh và chờ duyệt. Toàn bộ thí nghiệm N2a/N2b/N5/N3 (Module 3, 160 job) và
phân tích RQ1–RQ3 (`scripts/module3_rq_analysis.py`) đã chạy xong, dùng chung cho cả hai phương án.
Chưa sửa `main.tex`.

## 0. Khác biệt căn bản so với phương án TMLR

- **IJRCS cho phép bản mở rộng từ bài hội nghị một cách tường minh**: *"Manuscripts are accepted
  with the understanding that they are the original or extended versions of previously published
  papers in conferences"* (Author Guidelines, pubs2.ascee.org/index.php/ijrcs). Không cần tách bạch
  đóng góp như phương án TMLR.
- **Giới hạn trùng lặp có số cụ thể, không phải cấm tuyệt đối**: tổng đạo văn ≤25%, mỗi nguồn
  (kể cả chính bài ICERA) ≤4%. Nghĩa là được tóm tắt/diễn giải lại kết quả FB-tCNN+CCKD của ICERA,
  nhưng không được chép nguyên khối câu, bảng hay hình.
- **Khai báo chồng lấn trong "Comments to the Editor"** khi nộp, nêu rõ đây là bản mở rộng của
  ICERA 2026 #271 (đã chấp nhận), trích dẫn bài đó trong bài nộp.
- **Không thấy yêu cầu ẩn danh tác giả** trong checklist nộp bài — nhiều khả năng single-blind.
  Không cần viết ngôi thứ ba hay giấu link code như phương án TMLR.
- **Yêu cầu cấu trúc và số lượng khác hẳn**: tối thiểu 8 trang, có mục Literature Review, tối thiểu
  80 tài liệu tham khảo (gần như toàn bộ từ journal, trong 5 năm gần đây), tối thiểu 2 đóng góp nêu
  rõ trong Introduction, cấm tài liệu tham khảo do AI tạo.
- **Phạm vi "Robotics and Control Systems"** phù hợp hơn với phần triển khai phần cứng thời gian
  thực (RQ4, mục 6 bên dưới) so với phương án TMLR, nơi RQ4 chỉ là phần mở rộng tùy chọn. Ở đây RQ4
  nên là một trụ cột chính, không phải phụ lục.
- Tiền lệ thực tế: bài `qrd_eeg` (KD cho MI-EEG) của cùng tác giả đang revise tại chính IJRCS, cấu
  trúc thật là Introduction → Method → Results and Discussion → Conclusion (không tách mục Literature
  Review riêng — lồng vào Introduction), 106 tài liệu tham khảo. Dùng làm khuôn mẫu cấu trúc thực tế
  thay vì chỉ theo đúng mục lục liệt kê trên trang hướng dẫn.

## 1. Tiêu đề đề xuất

*Robust Cross-Channel Relational Knowledge Distillation for Low-Electrode SSVEP Brain–Computer
Interfaces: A Multi-Backbone Study with Real-Time Embedded Validation*

Lý do đổi tên so với bản TMLR: nhấn "brain-computer interface" và "real-time embedded validation"
để khớp phạm vi robotics/control, đồng thời giữ "robust" và "multi-backbone" làm từ khóa chính cho
RQ1–RQ3.

## 2. Abstract (bản nháp tiếng Anh — số liệu thật, không còn [TBD])

Toàn bộ số liệu dưới đây lấy trực tiếp từ `results/module3/analysis/*.csv`
(`scripts/module3_rq_analysis.py`, chạy 2026-11-01).

> Reducing the number of electrodes in a steady-state visual evoked potential (SSVEP)
> brain-computer interface improves wearability for applications such as wheelchair navigation and
> SSVEP-based spelling, but it degrades decoding accuracy. Cross-channel relational knowledge
> distillation (CCKD) recovers part of this loss by transferring the relational structure of a
> multi-electrode teacher to a low-electrode student. Prior work reported such gains for a single
> backbone, FB-tCNN. We evaluate CCKD on five public backbones (FB-tCNN, DeepConvNet,
> ShallowConvNet, EEGNet, and Compact-CNN), two 40-target datasets (Benchmark and BETA), and two
> student montages (three and one electrodes), each with five independent subject splits and eight
> seeds per split. At a fixed distillation weight, 5 of 20 backbone-dataset-montage settings
> improved significantly and 4 degraded significantly before correction; only 2 settings, both on
> EEGNet/BETA, survived Holm-Bonferroni correction, both negative. Evaluation on a single subject
> split misjudged the sign or significance of the effect in 10 of 20 settings, arguing against
> single-split evaluation for gains below 1-2 percentage points (pp). Swapping the student montage
> for an alternative electrode subset flipped the sign of the effect in 12 of 40 cases, showing the
> benefit is sensitive to electrode placement, not only electrode count. The accuracy gain
> correlated with how much CCKD increased linear CKA between teacher and student features (Pearson
> r = 0.55, p = 0.012; Spearman rho = 0.57, p = 0.009); CKA measured before distillation did not
> predict the gain (r = -0.11, p = 0.65). Distilling the last layer alone gave a larger
> gain than distilling an intermediate layer alone in only 5 of 20 settings; distilling both layers
> together matched or exceeded whichever single-layer choice was better in 8 of 20 settings. We
> further deploy the student models on a
> Raspberry Pi 4 and a Xilinx ZCU104, the hardware classes used in wheelchair-control and
> SSVEP-speller prototypes, and report [TBD: latency/power/accuracy after quantization, H1-H3 in
> Section 6]. These results show that cross-montage relational KD is backbone- and
> montage-dependent, that single-split evaluation is unreliable for the effect sizes typical of this
> literature, and that [TBD: embedded feasibility conclusion].

Ghi chú: hai [TBD] còn lại phụ thuộc thí nghiệm RQ4 (mục 6), chưa chạy. Phần còn lại không còn ô
trống nào — khác với bản TMLR trước đó.

## 3. Outline mới (theo cấu trúc IJRCS, tham chiếu cấu trúc thật của `qrd_eeg`)

1. **Introduction**
   - Bài toán giảm điện cực SSVEP, có đoạn mở bằng ứng dụng điều khiển thực tế: xe lăn điều khiển
     bằng SSVEP (electrode ít, phù hợp đội mũ hàng ngày) và bộ đánh vần (speller) SSVEP nhiều mục
     tiêu (electrode nhiều để phân biệt target nhưng cần gọn nhẹ cho dùng lâu dài). *Tài liệu tham
     khảo cho hai ứng dụng này CẦN TRA CỨU RIÊNG theo quy trình mục 3 AGENTS.md (Q1/Q2, ≤5 năm, DOI
     kiểm chứng qua skill `tim-dinh-huong-nghien-cuu`) — chưa có trích dẫn cụ thể trong bản kế hoạch
     này, không được tự điền.*
   - KD từ teacher nhiều kênh, kết quả đã biết trên FB-tCNN (trích ICERA 2026, khai báo đây là bản
     mở rộng).
   - Bốn câu hỏi nghiên cứu:
     - RQ1: Mức cải thiện có chuyển được sang backbone khác không?
     - RQ2: Đại lượng nào dự đoán mức cải thiện?
     - RQ3: Đánh giá trên một split có đáng tin không?
     - RQ4 (mới so với bản TMLR, đưa lên ngang hàng chứ không phụ): triển khai trên phần cứng nhúng
       dùng trong xe lăn/speller SSVEP có còn giữ được lợi ích của CCKD không?
   - Nêu rõ ít nhất 2 đóng góp (yêu cầu bắt buộc của IJRCS), ví dụ: (i) đánh giá có kiểm soát trên
     5 backbone × 2 dataset × 2 montage × 5 split × 8 seed kèm chỉ báo ∆CKA; (ii) kiểm chứng trên
     phần cứng nhúng thực tế (Pi 4, ZCU104) ở độ chính xác số học thấp.
2. **Literature Review** (có thể lồng vào cuối Introduction theo tiền lệ `qrd_eeg`, quyết định khi
   viết thật dựa trên độ dài)
   - Backbone học sâu cho SSVEP.
   - KD cho EEG, KD cross-input.
   - Ứng dụng điều khiển dựa trên SSVEP: xe lăn, speller, các ràng buộc phần cứng thực tế của chúng
     (số điện cực, độ trễ, công suất). *Cần tra cứu DOI riêng, xem ghi chú ở mục Introduction.*
   - Đo tương đồng biểu diễn (CKA).
   - Độ tin cậy đánh giá cross-subject trong EEG.
   - Lượng tử hóa/triển khai nhúng cho mô hình BCI thời gian thực.
   - Có thể trình bày dưới dạng bảng tổng hợp tài liệu (gợi ý trong hướng dẫn IJRCS).
3. **Method**
   - 3.1 CCKD tóm tắt kèm trích dẫn ICERA, một phương trình rút gọn.
   - 3.2 Năm backbone, bảng phân loại kiến trúc.
   - 3.3 Dataset và montage (gồm cả montage thay thế dùng cho RQ1b).
   - 3.4 Giao thức: 5 split × 8 seed; chọn λ theo validation riêng từng backbone; kiểm định (t một
     mẫu, Holm, TOST nếu cần).
   - 3.5 Triển khai nhúng: lượng tử hóa, đo trên Pi 4/ZCU104 (chi tiết ở mục 6).
4. **Results and Discussion**
   - 4.1 RQ1 — hiệu ứng theo backbone (bảng tổng hợp 20 ô, cột pHolm).
   - 4.2 RQ1b — độ nhạy montage (bảng 40 ô).
   - 4.3 RQ2 — ∆CKA dự đoán mức cải thiện, thí nghiệm can thiệp lớp distill.
   - 4.4 RQ3 — một split so với nhiều split, nondeterminism, chọn λ trên một split.
   - 4.5 RQ4 — kết quả nhúng, nối lại với ứng dụng xe lăn/speller đã nêu ở Introduction.
5. **Conclusions**
6. **References** (≥80, chủ yếu journal, ≤5 năm)
7. **Phụ lục**: bảng theo từng split, chi tiết siêu tham số, mã nguồn.

## 4. Bảng ánh xạ 26 bảng hiện có (so với phương án TMLR)

Vì giới hạn trùng lặp của IJRCS là theo tỷ lệ (25%/4%) chứ không cấm tuyệt đối, **nhiều bảng mục
TMLR đánh "Cắt" có thể đổi thành "Giữ (viết lại, không dùng câu/bảng gốc)"** — chỉ cần diễn giải lại
bằng số liệu riêng, không chép nguyên văn. Quyết định cuối cùng nên làm khi viết thật và chạy kiểm
tra đạo văn (similarity check) trước khi nộp, nhưng hướng chung:

| Bảng (theo đánh số TMLR) | Quyết định TMLR | Quyết định IJRCS | Lý do khác biệt |
|---|---|---|---|
| 2, 3 (FB-tCNN 1 split / 5 split) | Cắt | Giữ-PL, viết lại số liệu + diễn giải mới | Không bị cấm tuyệt đối tái dùng kết quả nếu diễn giải lại và trích dẫn rõ |
| 4, 7 (ablation chiến lược KD) | Cắt | Giữ-PL | Như trên, phục vụ phần Method làm rõ giao thức |
| 13, 14 (độ dài cửa sổ) | Cắt | Cắt | Lệch hẳn 4 RQ hiện tại, không cần dù có được phép |
| 25, 26 (Pi4/ITR bản ICERA) | Cắt | **Thay bằng đo lại trên cả Pi 4 và ZCU104 cho cả 5 backbone** | RQ4 giờ là trụ cột, không chỉ nhắc một câu |
| 1, 10, 11, 12, 16, 21, 22, 23, 24 | Giữ/Giữ-PL/Thay | Giữ nguyên quyết định | Không phụ thuộc nơi nộp |
| Các bảng còn lại | Cắt | Cắt | Lệch câu hỏi nghiên cứu ở cả hai phương án |

Bảng 21 (tổng hợp 5 backbone) **dùng số liệu RQ1 mới** (`rq1_backbone_effect_lsp100.csv`, không
phải số sơ bộ trong bản nháp abstract TMLR cũ): 5 ô dương/4 ô âm thô, 0 ô dương/2 ô âm sau Holm.

## 5. Tài liệu tham khảo (≥80, journal, ≤5 năm) — kế hoạch tra cứu

- Tiền lệ: bài `qrd_eeg` đạt 106 tài liệu tham khảo ở cùng tạp chí — khả thi.
- Nhóm cần bổ sung so với bản TMLR (vốn không giới hạn số lượng):
  1. SSVEP backbone/deep learning (có sẵn một phần từ bản thảo TMLR, cần bổ sung thêm bài journal
     2021–2026).
  2. KD cho EEG/BCI nói chung.
  3. **Ứng dụng điều khiển dựa trên SSVEP — xe lăn, speller, robot/tay máy** (nhóm mới, phục vụ
     phần mở đầu Introduction theo yêu cầu của người viết kế hoạch này). Chưa tra cứu.
  4. Lượng tử hóa/triển khai nhúng cho BCI thời gian thực.
  5. Đo tương đồng biểu diễn (CKA) và các chỉ báo tương tự.
  6. Độ tin cậy thống kê trong đánh giá EEG cross-subject.
- Không dùng công cụ sinh trích dẫn tự động (ChatGPT hay tương tự) theo đúng yêu cầu "cấm tài liệu
  tham khảo do AI tạo" của IJRCS — mọi DOI phải tự kiểm qua doi.org/Crossref như quy trình sẵn có.

### 5.1 Đã tra cứu (2026-11-01, skill `tim-dinh-huong-nghien-cuu` + NotebookLM deep research + Crossref)

81 bài, DOI đã kiểm qua Crossref (tiêu đề/tạp chí/năm khớp), nhóm theo mục ở trên. Vòng tra cứu đầu
(2026-11-01, sáng) có 27 bài; vòng bổ sung cùng ngày (chiều) thêm 54 bài để đạt mốc tối thiểu của
IJRCS. **Chưa kiểm hạng Q** qua scimagojr.com (trang chặn truy cập tự động) — cần tác giả tự xác nhận
trước khi đưa vào bài chính thức, nhất là các tạp chí chưa quen thuộc (Computers, Micromachines,
Entropy, BioMedInformatics, Bioengineering, Brain Sciences, Neurocomputing, IRBM, Biomedical Physics
& Engineering Express). Hai DOI nháp của NotebookLM bị sai một đoạn số (TNSRE.2021.9632600,
TNSRE.2020.9261605, neunet.2023.12.019, Journal of Mechanics in Medicine and Biology cũ,
BioMedEngOnLine cũ, J Neural Eng ear-electrode cũ) đã được sửa lại bằng tra cứu Crossref theo tiêu đề,
không dùng DOI do NotebookLM đưa ra khi DOI đó không phân giải được. Chi tiết trích dẫn đầy đủ: xem
notebook "SSVEP wheelchair + speller - IJRCS lit review" (`notebook.google.com`, hơn 80 nguồn đã nạp
qua nhiều lượt deep research + hỏi đáp lọc nguồn).

**Nhóm 3 — Ứng dụng điều khiển (wheelchair, speller, tay máy), 11 bài:**
1. Ahmed et al., "An SSVEP-Based BCI Device for Wheelchair Control Integrated with a Speech Aid System", *Eng* (MDPI), 2025, `10.3390/eng6120343`
2. Mai et al., "A hybrid BCI combining SSVEP and EOG and its application for continuous wheelchair control", *Biomedical Signal Processing and Control*, 2024, `10.1016/j.bspc.2023.105530`
3. Chen et al., "An Electric Wheelchair Manipulating System Using SSVEP-Based BCI System", *Biosensors* (MDPI), 2022, `10.3390/bios12100772`
4. Bai et al., "A hybrid P300-SSVEP BCI speller with a frequency enhanced row and column paradigm", *Frontiers in Neuroscience*, 2023, `10.3389/fnins.2023.1133933`
5. Wang et al., "A Methodology for Enhancing SSVEP Features Using Adaptive Filtering Based on the Spatial Distribution of EEG Signals", *Micromachines* (MDPI), 2023, `10.3390/mi14050976`
6. Sakkalis, Krana, Farmaki, Pediaditis, "Augmented Reality Driven Steady-State Visual Evoked Potentials for Wheelchair Navigation", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2022, `10.1109/TNSRE.2022.3215695`
7. Liu, Yu, Liu, Tang, Liang, Chu, Zhou, "A novel brain-controlled wheelchair combined with computer vision and augmented reality", *BioMedical Engineering OnLine*, 2022, `10.1186/s12938-022-01020-8`
8. Rivera-Flor, Gurve, Floriano, Delisle-Rodriguez, Mello, Bastos-Filho, "CCA-based compressive sensing for SSVEP-based brain-computer interfaces to command a robotic wheelchair", *IEEE Trans. Instrumentation and Measurement*, 2022, `10.1109/TIM.2022.3218102`
9. Albán-Escobar, Navarrete-Arroyo, De la Cruz-Guevara, Tobar-Quevedo, "Assistance Device Based on SSVEP-BCI Online to Control a 6-DOF Robotic Arm", *Sensors* (MDPI), 2024, `10.3390/s24061922`
10. Naser, Bhattacharya, "Towards Practical BCI-Driven Wheelchairs: A Systematic Review Study", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2023, `10.1109/tnsre.2023.3236251`
11. Ukaegbu, Houshmand, Hammond, Adams, Andersen, Rouhani, "Navigation paradigms for non-invasive BCI-controlled wheelchairs: a systematic review", *Progress in Biomedical Engineering*, 2026, `10.1088/2516-1091/ae5563`

**Nhóm 1 — Backbone SSVEP / giải mã sâu, kể cả độ nhạy số điện cực và tổng quan chung, 35 bài:**
12. He et al., "TBMSCCN: Two-Branch Multi-Scale Convolutional Correlation Network for SSVEP Classification", *IEEE Trans. Biomedical Engineering*, 2026, `10.1109/TBME.2026.3676014`
13. Lan et al., "MTSNet: Convolution-Based Transformer Network ... for SSVEP Signal Decoding", *IEEE J. Biomedical and Health Informatics*, 2025, `10.1109/JBHI.2025.3573410`
14. Song et al., "EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2023, `10.1109/TNSRE.2022.3230250`
15. Chen, Zhang, Pan, Xu, Guan, "A Transformer-based deep neural network model for SSVEP classification (SSVEPformer)", *Neural Networks*, 2023, `10.1016/j.neunet.2023.04.045`
16. Ding, Shan, Fang, Wang, Sun, Li, "Filter Bank Convolutional Neural Network for Short Time-Window Steady-State Visual Evoked Potential Classification (FB-tCNN)", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2021, `10.1109/TNSRE.2021.3132162` — **backbone gốc dùng trong bản ICERA/CCKD**
17. Li, Xiang, Kesavadas, "Convolutional Correlation Analysis for Enhancing the Performance of SSVEP-Based Brain-Computer Interface (ConvCA)", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2020, `10.1109/TNSRE.2020.3038718`
18. Zhang, An, Liu, Chen, Wei, Sun, "Dynamic decomposition graph convolutional neural network for SSVEP-based brain–computer interface (DDGCNN)", *Neural Networks*, 2024, `10.1016/j.neunet.2023.12.029`
19. Pan, Chen, Zhang, Zhang, "An efficient CNN-LSTM network with spectral normalization and label smoothing technologies for SSVEP frequency recognition (SSVEPNet)", *Journal of Neural Engineering*, 2022, `10.1088/1741-2552/ac8dc5`
20. Wang, Wong, Wang, Feng, Cong, Wan, "Compact Artificial Neural Network Based on Task Attention for Individual SSVEP Recognition with Less Calibration", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2023, `10.1109/TNSRE.2023.3276745`
21. Huang, Wei, "A Deep Learning Model Combining Convolutional Neural Networks and a Selective Kernel Mechanism for SSVEP-Based BCIs", *Computers in Biology and Medicine*, 2025, `10.1016/j.compbiomed.2025.110691`
22. Israsena, Pan-Ngum, "A CNN-Based Deep Learning Approach for SSVEP Detection Targeting Binaural Ear-EEG", *Frontiers in Computational Neuroscience*, 2022, `10.3389/fncom.2022.868642`
23. Huang, Chen, Cao, Cao, Yang, Dong, Liu, "IncepFormerNet: A multi-scale multi-head attention network for SSVEP classification", *Journal of Mechanics in Medicine and Biology*, 2025, `10.1142/S0219519425400809`
24. Wei, Li, Wang, Gao, "Enhancing the performance of SSVEP-based BCIs by combining task-related component analysis and deep neural network", *Scientific Reports*, 2025, `10.1038/s41598-024-84534-6`
25. Li, Yang, Fei, Wang, Huang, Hu, "A Convolutional Neural Network for SSVEP Identification by Using a Few-Channel EEG", *Bioengineering* (MDPI), 2024, `10.3390/bioengineering11060613`
26. Ma, Cao, Xie, Zhang, Li, Zhang, "LGFCNN: A Synergistic Framework Integrating Graph-Based Spatial Filter and Lightweight CNN for SSVEP Recognition", *Neurocomputing*, 2025, `10.1016/j.neucom.2025.131561`
27. Shen, Ke, Ming, "Enhancing SSVEP recognition for short data via time series forecasting", *Expert Systems with Applications*, 2026, `10.1016/j.eswa.2025.130640`
28. Ji, Xu, Chen, Yin, Wan, Wang, "Subject-specific CNN model with parameter-based transfer learning for SSVEP detection", *Biomedical Signal Processing and Control*, 2025, `10.1016/j.bspc.2024.107404`
29. Oikonomou, "MultiHeadEEGModelCLS: Contextual Alignment and Spatio-Temporal Attention Model for EEG-Based SSVEP Classification", *Electronics* (MDPI), 2025, `10.3390/electronics14224394`
30. Wen, Jia, Han, Dong, Gao, Cao, Hao, Guo, Cao, "Filter banks guided correlational convolutional neural network for SSVEPs based BCI classification", *Journal of Neural Engineering*, 2024, `10.1088/1741-2552/ad7f89`
31. Xiong, Song, Liu, Han, "Deep transfer learning-based SSVEP frequency domain decoding method", *Biomedical Signal Processing and Control*, 2024, `10.1016/j.bspc.2023.105931`
32. Yang, Zhao, Zhang, Liu, "High-Order Temporal Convolutional Network for Improving Classification Performance of SSVEP-EEG", *IRBM*, 2024, `10.1016/j.irbm.2024.100830`
33. Xia, Wang, Xiong, Deng, "Enhancing detection of SSVEP-based BCIs via a novel temporally local canonical correlation analysis", *Journal of Neuroscience Methods*, 2025, `10.1016/j.jneumeth.2024.110325`
34. Hamou, Moufassih, Tarahi, Agounad, Azami, "Hybrid approach: combining eCCA and SSCOR for enhancing SSVEP decoding", *The Journal of Supercomputing*, 2024, `10.1007/s11227-024-06027-7`
35. Hu, Wang, Tu, Guo, Yan, "Cross-domain correlation analysis to improve SSVEP signals recognition in brain-computer interfaces", *Biomedical Physics & Engineering Express*, 2026, `10.1088/2057-1976/ae2772`
36. Kim, Im, "Influence of the Number of Channels and Classification Algorithm on the Performance Robustness to Electrode Shift in SSVEP-BCI", *Frontiers in Neuroinformatics*, 2021, `10.3389/fninf.2021.750839`
37. Kiser, Cantürk, Volosyak, "SSVEP-driven BCI authentication with reduced number of EEG electrodes across high and low frequency ranges", *Frontiers in Neuroergonomics*, 2026, `10.3389/fnrgo.2026.1741655`
38. Zhao, Zheng, Yuan, Pei, "Optimization of ear electrodes for SSVEP-based BCI", *Journal of Neural Engineering*, 2023, `10.1088/1741-2552/acdf85`
39. Liu, Zhang, Liu, Liu, Ming, Liu, Suo, Funahashi, Yan, "Review of brain–computer interface based on steady-state visual evoked potential", *Brain Science Advances*, 2022, `10.26599/BSA.2022.9050022`
40. Albahri, Al-qaysi, Alzubaidi, Alnoor, Albahri, Alamoodi, Bakar, "A Systematic Review of Using Deep Learning Technology in the Steady-State Visually Evoked Potential-Based Brain-Computer Interface Applications", *International Journal of Telemedicine and Applications*, 2023, `10.1155/2023/7741735`
41. Al-Qaysi, Albahri, Ahmed, Hamid, Alsalem, Albahri, Alamoodi, Homod, Shayea, Duhaim, "A Comprehensive Review of Deep Learning Power in Steady-State Visual Evoked Potentials", *Neural Computing and Applications*, 2024, `10.1007/s00521-024-10143-z`
42. Reitelbach, Oyibo, "Optimal Stimulus Properties for Steady-State Visually Evoked Potential Brain–Computer Interfaces: A Scoping Review", *Multimodal Technologies and Interaction* (MDPI), 2024, `10.3390/mti8020006`
43. Kawala-Sterniuk, Browarska, Al-Bakri, Pelc, Zygarlicki, Sidikova, Gorzelańczyk, "Summary of over Fifty Years with Brain–Computer Interfaces — A Review", *Brain Sciences* (MDPI), 2021, `10.3390/brainsci11010043`
44. Zhang, Yao, Wang, Monaghan, McAlpine, Zhang, "A survey on deep learning-based non-invasive brain signals: recent advances and new frontiers", *Journal of Neural Engineering*, 2021, `10.1088/1741-2552/abc902`
45. Abibullaev, Keutayeva, Zollanvari, "Deep Learning in EEG-Based BCIs: A Comprehensive Review of Transformer Models, Advantages, Challenges, and Applications", *IEEE Access*, 2023, `10.1109/access.2023.3329678`
46. Karikari, Koshechkin, "Review on brain-computer interface technologies in healthcare", *Biophysical Reviews*, 2023, `10.1007/s12551-023-01138-6`

**Nhóm 2 — Knowledge distillation cho EEG/BCI, 11 bài:**
47. Huang, Chen, Wei, "Enhancing Low-Density EEG-Based BCI with Similarity-Keeping Knowledge Distillation", *IEEE Trans. Emerging Topics in Computational Intelligence*, 2024, `10.1109/TETCI.2023.3335943` — **tiền lệ gần nhất với CCKD** (similarity-preserving KD cho EEG mật độ điện cực thấp)
48. Zheng et al., "Cross-Subject Motor Imagery EEG Decoding with Domain Generalization", *Bioengineering* (MDPI), 2025, `10.3390/bioengineering12050495`
49. Xu & Yu, "Entropy-Based Dual-Teacher Distillation for Efficient Motor Imagery EEG Classification", *Entropy* (MDPI), 2026, `10.3390/e28030310`
50. Zhang & Lu, "Multimodal Knowledge Distillation for Emotion Recognition", *Brain Sciences* (MDPI), 2025, `10.3390/brainsci15070707`
51. Zhang, Tang, Guan, "Visual-to-EEG cross-modal knowledge distillation for continuous emotion recognition", *Pattern Recognition*, 2022, `10.1016/j.patcog.2022.108833`
52. Cheng et al., "DISD-Net: A Dynamic Interactive Network With Self-Distillation for Cross-Subject Multi-Modal Emotion Recognition", *IEEE Trans. Multimedia*, 2025, `10.1109/TMM.2025.3535344`
53. Wu et al., "A Multi-level Teacher Assistant-based KD Framework with Dynamic Feedback for Motor Imagery EEG Decoding", *Neural Networks*, 2026, `10.1016/j.neunet.2025.108180`
54. Zhang, Liu, Xiao, Quek, "LLM-Enhanced Multi-Teacher Knowledge Distillation for Modality-Incomplete Emotion Recognition in Daily Healthcare", *IEEE Journal of Biomedical and Health Informatics*, 2025, `10.1109/JBHI.2024.3470338`
55. Zhang, Chen, Cai, Cheng, "SASD-MCL: Semi-supervised alignment self-distillation with mixed contrastive learning for cross-subject EEG emotion recognition", *Neural Networks*, 2025, `10.1016/j.neunet.2025.107614`
56. Hua Zhong, Shi, Wang, "Knowledge-guided quantization-aware training for EEG-based emotion recognition", *Journal of Visual Communication and Image Representation*, 2025, `10.1016/j.jvcir.2025.104415` — nối giữa nhóm 2 và nhóm 4 (KD kèm lượng tử hóa)
57. Li, Wang, Shao, Huang, "Distillation-Based Domain Generalization for Cross-Dataset EEG-Based Emotion Recognition", *IEEE Trans. Emerging Topics in Computational Intelligence*, 2025, `10.1109/TETCI.2024.3449926`

**Nhóm 4 — Lượng tử hóa / triển khai nhúng, 12 bài:**
58. Xie, Majoros, Oniga, "A Comprehensive Review of Hardware Acceleration Techniques and CNN for EEG Signals", *Sensors* (MDPI), 2024, `10.3390/s24175813`
59. Mullane, Przychodni, Crispin-Bailey, "Real-Time EEG-Based Anaesthesia Monitoring Using Neural Networks on Edge-AI Hardware", *BioMedInformatics* (MDPI), 2026, `10.3390/biomedinformatics6050080`
60. Pacini et al., "Design and Evaluation of CPU-, GPU-, and FPGA-Based Deployment of a CNN for Motor Imagery Classification in BCI", *Electronics* (MDPI), 2024, `10.3390/electronics13091646`
61. Nguyen, Do, Tran, Nguyen, Lin, "Edge AI-Brain–Computer Interfaces System: A Survey", *IEEE Trans. Neural Systems and Rehabilitation Engineering*, 2025, `10.1109/TNSRE.2025.3618688`
62. Palumbo, Vizza, Calabrese, Ielpo, "An FPGA-Embedded BCI System to Support Individual Autonomy in Locked-In Individuals", *Sensors* (MDPI), 2022, `10.3390/s22010318`
63. Novac, Boukli Hacene, Pegatoquet, Miramond, Gripon, "Quantization and deployment of deep neural networks on microcontrollers", *Sensors* (MDPI), 2021, `10.3390/s21092984`
64. Chu, Yan, Gan, Jia, Qian, Huan, Zheng, Zou, "A Neuromorphic Processing System With Spike-Driven SNN Processor for Wearable ECG Classification", *IEEE Trans. Biomedical Circuits and Systems*, 2022, `10.1109/TBCAS.2022.3189364`
65. Zhao, Yang, Wang, Fang, Liu, Zhang, Sawan, "A 0.99-to-4.38 μJ/class Event-Driven Hybrid Neural Network Processor for Full-Spectrum Neural Signal Analyses", *IEEE Trans. Biomedical Circuits and Systems*, 2023, `10.1109/TBCAS.2023.3268502`
66. Busia, Leone, Matticola, Raffo, Meloni, "Wearable Epilepsy Seizure Detection on FPGA With Spiking Neural Networks", *IEEE Trans. Biomedical Circuits and Systems*, 2025, `10.1109/TBCAS.2025.3575327`
67. Scrugli, Leone, Busia, Raffo, Meloni, "All-Spiking ECG Analysis for Arrhythmia Classification on Low-Power FPGA", *IEEE Sensors Journal*, 2026, `10.1109/JSEN.2026.3663715`
68. Yuan, Wei, Liu, "Spiking neural networks for EEG signal analysis using wavelet transform", *Frontiers in Neuroscience*, 2025, `10.3389/fnins.2025.1652274`
69. Aslam, Altaf, "An On-Chip Processor for Chronic Neurological Disorders Assistance Using Negative Affectivity Classification", *IEEE Trans. Biomedical Circuits and Systems*, 2020, `10.1109/TBCAS.2020.3008766`

**Nhóm 5 — CKA, 5 bài (3 bài chính sau 2020 + 2 bài nền tảng trước 2020):**
70. Tobon-Henao, Álvarez-Meza, Castellanos-Dominguez, "Kernel-Based Regularized EEGNet Using Centered Alignment and Gaussian Connectivity for Motor Imagery Discrimination", *Computers* (MDPI), 2023, `10.3390/computers12070145` — CKA áp dụng trực tiếp cho EEGNet
71. Alvarez, "Gaussian RBF Centered Kernel Alignment (CKA) in the Large-Bandwidth Limit", *IEEE Trans. Pattern Analysis and Machine Intelligence*, 2023, `10.1109/TPAMI.2022.3216518`
72. Kriegeskorte, Wei, "Neural tuning and representational geometry", *Nature Reviews Neuroscience*, 2021, `10.1038/s41583-021-00502-3`
73. Kornblith, Norouzi, Lee, Hinton, "Similarity of Neural Network Representations Revisited", *Proc. ICML 2019* (PMLR 97:3519-3529) — **(bài nền tảng, trước 2020)**, định nghĩa gốc của CKA
74. Cortes, Mohri, Rostamizadeh, "Algorithms for Learning Kernels Based on Centered Alignment", *JMLR* 13:795-828, 2012 — **(bài nền tảng, trước 2020)**

**Nhóm 6 — Độ tin cậy thống kê đánh giá cross-subject EEG, trực tiếp phục vụ RQ3, 7 bài:**
75. White, "k-Fold Cross-Validation Can Significantly Over-Estimate True Classification Accuracy in Common EEG-Based Passive BCI Experimental Designs", *Sensors* (MDPI), 2023, `10.3390/s23136077`
76. Del Pup et al., "The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject analysis", *Computers in Biology and Medicine*, 2025, `10.1016/j.compbiomed.2025.110608`
77. Brookshire, Kasper, Blauch, Lucero, "Data leakage in deep learning studies of translational EEG", *Frontiers in Neuroscience*, 2024, `10.3389/fnins.2024.1373515`
78. Lee, Cheon, Lee, Shim, Hwang, "Risk of data leakage in estimating the diagnostic performance of a deep-learning-based computer-aided system for psychiatric disorders", *Scientific Reports*, 2023, `10.1038/s41598-023-43542-8`
79. Kamrud, Borghetti, Schubert Kabban, "The Effects of Individual Differences, Non-Stationarity, and the Importance of Data Partitioning Decisions for Training and Testing of EEG Cross-Participant Models", *Sensors* (MDPI), 2021, `10.3390/s21093225`
80. Xu, Xu, Ke, An, Liu, Ming, "Cross-Dataset Variability Problem in EEG Decoding With Deep Learning", *Frontiers in Human Neuroscience*, 2020, `10.3389/fnhum.2020.00103`
81. Geethapriya, Pattabiraman, "Subject-independent classification of ADHD from pediatric EEG: a leakage-aware deep learning framework with interpretable neurophysiological mapping", *Frontiers in Neuroscience*, 2026, `10.3389/fnins.2026.1925266`

**Việc còn lại:** 81/≥80 — đã đạt mốc tối thiểu về số lượng. Còn hai việc trước khi dùng chính thức:
(i) kiểm hạng Q qua scimagojr.com thủ công cho toàn bộ 81 bài, nhất là các tạp chí ít quen (xem danh
sách ở trên); (ii) rà lại độ phù hợp của từng bài với đúng câu hỏi nghiên cứu khi viết Related Work
thật — danh sách này ưu tiên số lượng và DOI đúng, một số bài (ví dụ nhóm KD tổng quát cho cảm xúc,
không phải SSVEP) chỉ nên dùng để lập luận chung, không nên trích dẫn như tiền lệ trực tiếp.

## 6. RQ4: phần cứng nhúng — nâng vai trò so với bản TMLR

Giữ nguyên nội dung kỹ thuật đã kiểm ở `TMLR_pivot_plan.md` mục 6.1–6.3 (rào cản kernel/activation
của DPU, hướng HLS khả thi hơn, hạ tầng `ubuntu-huy`/ZCU104 đã thông). Khác biệt ở đây:

- H1–H3 (lượng tử hóa INT8, mô phỏng fixed-point hls4ml, đo độ trễ/bộ nhớ Pi 4 + ZCU104) chuyển từ
  "bắt buộc nhưng phụ" sang **nội dung chính của Section 4.5 và một phần Method 3.5**, không phải
  phụ lục.
- H4 (tổng hợp PL đầy đủ trên ZCU104) nên làm, không chỉ "tùy chọn" — vì đây là bằng chứng khả thi
  mạnh nhất cho một tạp chí Robotics and Control Systems, và hạ tầng đã có sẵn từ `ssvep_lite`
  (NCS Đào Ngọc Tôn, cần thống nhất việc dùng lại và ghi công, như đã nêu trong bản TMLR).
- Khung ứng dụng: nối kết quả độ trễ/công suất với yêu cầu thời gian thực của xe lăn (lệnh điều
  khiển phải ra trong khoảng trăm mili-giây để an toàn) và speller (tốc độ ký tự/phút phụ thuộc độ
  trễ pipeline) — đúng phần người viết kế hoạch này yêu cầu thêm. Số liệu ITR đã có từ Module 2 có
  thể tái dùng có trích dẫn, không cần đo lại riêng cho khung này.

### 6.1 Kế hoạch triển khai chi tiết (2026-11-01, chờ duyệt trước khi chạy)

Theo đúng quy tắc "toán học hóa trước khi chạy" (memory.md, mục Quy trình): mỗi hạng mục dưới đây có
mô hình dự đoán định lượng và tiêu chí GO/NO-GO khóa trước, chỉ chạy để kiểm mô hình, không quét
tham số vô định. Sau khi có kết quả, rà soát lại một lượt (đúng yêu cầu của anh) trước khi viết vào
Mục 4.5/3.5 của bản thảo.

**Rào cản hạ tầng cần giải quyết trước — chưa chạy được job nào nếu chưa xong các mục này:**

| # | Việc | Trạng thái | Cần |
|---|---|---|---|
| 1 | Pi 4 chưa cấu hình SSH qua Tailscale | **Xong (2026-11-01)** — xác nhận `ssh pi4` trả về `huy`, 4 nhân | — |
| 2 | `hls4ml` chưa cài trên `ubuntu-huy` | **Xong (2026-11-01)** — đã có sẵn qua env conda `hls4ml_cd2` (hls4ml 1.3.0) của NCS Đào Ngọc Tôn, không cần cài mới | — |
| 3 | Vitis/Vitis HLS 2024.2 trên `ubuntu-huy` | **Xong (2026-11-01)** — xác minh lại, vẫn còn nguyên tại `/tools/Xilinx` | — |
| 4 | Export checkpoint PyTorch (N2b/N3) sang ONNX | Chưa có script | Viết `scripts/export_onnx.py`, kiểm đủ 5 kiến trúc chạy qua `torch.onnx.export` không lỗi (ELU, square/log của ShallowConvNet cần kiểm riêng) |
| 5 | Thống nhất với NCS Đào Ngọc Tôn về dùng lại khung HLS G3.1 của `ssvep_lite` cho H4 | **Xong (2026-11-01)** — đã trao đổi, được đồng ý | — |

**H1 — INT8 PTQ trên CPU (bắt buộc).**
Phạm vi: dùng lại checkpoint đã có của N2b (20 ô gốc: 5 backbone × 2 dataset × 2 montage, λ theo
Bảng λ đã chọn), không huấn luyện lại — chỉ lượng tử hóa tĩnh (PTQ, hiệu chuẩn trên tập validation)
rồi đánh giá trên test, giữ nguyên 5 split × 8 seed để còn kiểm định thống kê được như RQ1.
Mô hình dự đoán: gọi $\delta_{\text{PTQ}}$ là nhiễu do lượng tử hóa trên mỗi lần chạy; theo tài liệu
đã khảo sát (R63 Novac và cộng sự và các bài INT8 khác ở mục 2.6), PTQ tĩnh không retrain cho mạng
nhỏ cỡ này thường gây lệch dưới 1~pp. Dự đoán định lượng: $|\delta_{\text{PTQ}}|$ cỡ 0,1–0,5~pp,
cùng bậc với nhiễu không xác định đã đo ở Mục 4.4 (0,3–0,5~pp).
Tiêu chí GO: dấu của $\Delta_{\text{INT8}}$ khớp dấu $\Delta_{\text{FP32}}$ ở phần lớn trong 2 tổ hợp
còn ý nghĩa sau Holm và các tổ hợp có ý nghĩa thô; $|\Delta_{\text{INT8}} - \Delta_{\text{FP32}}|$
nằm trong khoảng nhiễu nền đã đo. NO-GO/cần xem lại mô hình nếu lượng tử hóa đảo dấu một cách hệ
thống (trên nửa số tổ hợp) hoặc lệch lớn hơn nhiều lần nhiễu nền.
Đầu ra: `results/module3/h1_int8/`.

**H2 — Mô phỏng fixed-point bằng hls4ml (bắt buộc, phạm vi thu hẹp).**
Phạm vi: không lặp lại cả 20 ô. Chỉ chọn backbone nhỏ nhất (Compact-CNN hoặc EEGNet, theo số tham số
ở Bảng 1) và lớn nhất (FB-tCNN, do kernel thời gian 250 tốn tài nguyên nhất) × một dataset (BETA) ×
quét độ rộng bit $\{16, 12, 10, 8\}$. Mục tiêu là tìm "điểm gãy" độ rộng bit, không phải lặp lại RQ1.
Mô hình dự đoán: độ chính xác giảm đơn điệu khi giảm bit, điểm gãy dự kiến quanh 8–10 bit (theo tài
liệu lượng tử hóa CNN phổ biến). Dự đoán định lượng: giữ trong 1~pp của FP32 ở $\geq 10$ bit; dưới 8
bit có thể giảm mạnh ($>2$~pp).
Tiêu chí GO: điểm gãy đo được rơi trong khoảng dự đoán (8–10 bit) — xác nhận mô hình đúng, dùng mức
bit đó cho H3/H4. Nếu lệch nhiều (ví dụ gãy ở 14 bit), chỉ quét thêm các điểm quanh đó, không mở
rộng toàn bộ lưới.
Đầu ra: `results/module3/h2_fixedpoint/`.

**H3 — Độ trễ và bộ nhớ trên Pi 4 và ZCU104 PS (bắt buộc).**
Phạm vi: 5 backbone × 2 mức độ chính xác (FP32, INT8 ở mức bit chọn từ H2) × 2 board, batch 1, cửa
sổ 1,0~s, lặp lại suy luận (ví dụ 50–100 lần) lấy trung bình và độ lệch chuẩn, theo đúng giao thức đo
Pi 4 đã dùng ở bài ICERA trước.
Mô hình dự đoán: độ trễ suy luận tỷ lệ thuận với số tham số/FLOPs của backbone (Bảng 1); dự đoán thứ
hạng độ trễ trước khi đo: Compact-CNN và EEGNet (depthwise) nhanh nhất, FB-tCNN (kernel 250) chậm
nhất. Đo xong đối chiếu đúng thứ hạng dự đoán không, không chỉ đo rồi báo số.
Tiêu chí GO (liên hệ trực tiếp ứng dụng xe lăn/bộ đánh vần đã nêu ở Introduction): độ trễ suy luận
cộng với cửa sổ 1,0~s phải nằm trong ngân sách vòng điều khiển thời gian thực đã nêu (cỡ vài trăm
mili-giây cho lệnh xe lăn). Nếu mọi backbone đạt → kết luận "khả thi thời gian thực"; nếu backbone
lớn (FB-tCNN) vượt ngưỡng, nêu rõ đánh đổi trong Thảo luận, không che giấu.
Đầu ra: `results/module3/h3_latency/`.

**H4 — Tổng hợp PL đầy đủ trên ZCU104 (bằng chứng khả thi, không phải phụ lục).**
Phạm vi: một backbone đại diện, nhỏ và khả thi tài nguyên nhất theo Bảng 1 (dự kiến EEGNet hoặc
Compact-CNN, chốt sau khi có H2). Tái dùng khung Vitis HLS → Vivado → overlay PYNQ đã có sẵn của
`ssvep_lite` (G3.1, `~/ssvep_lite_hls_g31` trên `ubuntu-huy`) thay vì đi đường DPU (đã loại ở mục
6.1 bản TMLR vì giới hạn kernel/activation). Đo LUT/DSP/BRAM, công suất qua `ina226`/`irps5401`, và
xác minh độ chính xác đo được trên board so với mô phỏng bit-accurate ở H2.
Không cần thống kê qua nhiều split — đây là một lần tổng hợp để chứng minh khả thi, không phải thí
nghiệm thống kê.
Điều kiện tiên quyết: xong rào cản hạ tầng #2, #3, #5 ở trên.
Đầu ra: `results/module3/h4_hls/`.

**Việc tiếp theo:** anh duyệt phạm vi (đặc biệt H2/H4 đã thu hẹp để không chạy vô định), rồi tôi giải
quyết rào cản hạ tầng theo thứ tự #1 → #4 trước, báo lại từng bước thay vì làm hết một lúc. H3/H4 cần
truy cập phần cứng thật, không chạy ngầm được — sẽ xin xác nhận thời gian cụ thể trước khi đo.

## 7. Tuân thủ nguyên gốc khi nộp IJRCS

- Khai báo trong "Comments to the Editor" lúc nộp: đây là bản mở rộng của ICERA 2026 #271 (đã chấp
  nhận), kèm trích dẫn đầy đủ trong bài.
- Trước khi nộp, chạy kiểm tra trùng lặp (Turnitin/iThenticate hoặc công cụ tương đương của trường)
  so với bản PDF ICERA, đảm bảo tổng thể ≤25% và riêng so với ICERA ≤4%. Nếu vượt ngưỡng, viết lại
  thêm phần Method/Results liên quan đến FB-tCNN.
- Không dùng lại Algorithm/Figure gốc của ICERA dưới dạng ảnh chụp/sao chép — vẽ lại nếu cần minh
  họa tương tự.
- Kiểm tra điều khoản bản quyền của ICERA/TNUT trước khi trích dẫn số liệu hoặc hình bất kỳ.

## 8. Trạng thái quyết định

- [x] **Chọn IJRCS** (2026-11-01, xác nhận trực tiếp). `TMLR_pivot_plan.md` giữ lại làm tài liệu
  tham khảo, không tiếp tục phát triển song song trừ khi có quyết định mới.
- [x] Duyệt tiêu đề mục 1 và outline mục 3 (ngầm định qua việc triển khai bản thảo thật, xem mục 9).
- [ ] Duyệt khung RQ4 nâng vai trò ở mục 6 (kéo theo khối lượng công việc phần cứng lớn hơn hẳn
  phương án TMLR — cần xác nhận còn đủ thời gian/nhân lực cho H1–H4). Phần Method/Result của RQ4
  trong bản thảo vẫn để `[TBD]`, chưa chạy thí nghiệm.
- [x] Tra cứu tài liệu tham khảo (mục 5) hoàn tất, 81 bài, xem mục 5.1.
- [x] Thống nhất với NCS Đào Ngọc Tôn về việc dùng lại luồng HLS của `ssvep_lite` cho H4 (2026-11-01,
  đã trao đổi, được đồng ý).

## 9. Bản thảo thật đã dựng

- Thư mục: `paper/ijrcs/` (tách khỏi `paper/latex/` của bản TMLR, không ảnh hưởng bản đó).
  - `IJRCS_CCKD_SSVEP.tex`: dựng từ `paper/IJRCS template.zip` (mẫu thật của tạp chí), đã biên dịch
    sạch bằng `pdflatex` (11 trang, xem `IJRCS_CCKD_SSVEP.pdf`). Cài bổ sung các gói LaTeX còn thiếu
    trong TinyTeX của tài khoản này qua `tlmgr` (titlesec, caption, newtx, babel-english và các gói
    phụ thuộc) — chỉ ảnh hưởng `~/.TinyTeX` của `gv_bvnam`, không phải thao tác chung cho cả cụm.
  - `references.tex`: 82 `\bibitem` (81 tài liệu ở mục 5.1 + 1 mục trích dẫn bài ICERA 2026 #271 đang
    chờ xuất bản, dạng "accepted, in press"), kiểu IEEE, đầy đủ tên tác giả (không dùng "et al." theo
    đúng yêu cầu mẫu), DOI đã kiểm qua Crossref ở mục 5.1. Hai tài liệu nền tảng CKA trước 2020
    (Kornblith 2019, Cortes 2012) không có DOI phân giải được nên dùng link cố định
    (`proceedings.mlr.press`, `jmlr.org`) thay DOI, theo đúng kiểu mẫu tạp chí cho phép (xem ví dụ
    R3 trong `IJRCS template.zip`).
  - Đã viết: Title, Abstract (số liệu thật từ `results/module3/analysis/*.csv`, còn 2 chỗ `\tbd{}`
    cho RQ4 chưa chạy), Keywords, Introduction (7 đoạn, đủ ≥1 ref vấn đề + ≥2 ref giải pháp + ≥5 ref
    tổng quan + 4 đóng góp theo yêu cầu mẫu), Related Work/Literature Review (6 tiểu mục theo 6 nhóm
    tài liệu ở mục 5.1, kết bằng tiểu mục tổng hợp khoảng trống nghiên cứu).
  - Khung Introduction/Related Work đã viết lại (2026-11-01, theo yêu cầu "viết phù hợp hơn với
    hướng điều khiển của tạp chí") để nhấn khung vòng điều khiển kín: BCI là khối sinh lệnh điều
    khiển, độ trễ quyết định là một phần độ trễ vòng lặp, ITR là băng thông kênh điều khiển, số điện
    cực/độ phức tạp mô hình ảnh hưởng số bậc tự do (DOF) điều khiển được an toàn. Áp dụng nhất quán ở
    đoạn mở Introduction, mục 2.3 (ứng dụng điều khiển), mục 2.6 (nhúng/lượng tử hóa) và mục 2.7
    (tổng hợp khoảng trống).
  - Chưa viết: Method, Result and Discussion, Conclusion (đánh dấu `\tbd{}`, chờ chạy RQ4 và chốt
    số liệu RQ1–RQ3 vào bảng thật). Author Contribution cũng để `\tbd{}` vì cần tác giả tự xác nhận
    phần đóng góp cụ thể của từng người, không tự suy đoán.
