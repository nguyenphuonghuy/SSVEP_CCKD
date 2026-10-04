# Rà soát code và đối chiếu bài báo — 03/10/2026

Phạm vi: paper/latex/main.tex, datasets/public_ssvep.py, models/fbtcnn.py, models/cckd_backbones.py, training/*.py, utils/config.py, configs/*.yaml, tests/smoke_test.py, deploy/pi4_latency/cckd_latency_pi.py và các JSON thí nghiệm chính. Không chỉnh sửa code huấn luyện hoặc dữ liệu/kết quả hiện có. Không chạy lại huấn luyện đầy đủ hoặc benchmark trên Raspberry Pi.

## 1. Sai lệch công thức CCKD giữa bài báo và code — ưu tiên cao

Bài báo main.tex:152–168 và thuật toán mô tả G/||G||_F cho toàn ma trận, loss là tổng bình phương sai khác. training/kd_losses.py:49–52 dùng F.normalize(..., dim=1), tức chuẩn hóa L2 từng hàng, rồi chia loss cho B². Đây là hai mục tiêu khác nhau, không chỉ khác hệ số hằng. Ví dụ đặc trưng student [[1,2],[3,1],[2,4]], teacher [[2,1],[1,3],[1,2]]: code cho 0.0458316, công thức bài báo cho 0.2546766.

Code còn lấy trung bình loss qua các tầng (kd_losses.py:119), tương ứng alpha_l=1/L; bài báo chưa nêu giá trị alpha. Nên sửa mô tả và thuật toán theo implementation đã tạo kết quả, hoặc nếu quyết định đổi implementation theo bài báo thì phải huấn luyện lại và dò lambda. Không được giữ các bảng cũ rồi thay loss âm thầm.

## 2. Resume có thể trộn kết quả từ cấu hình khác — ưu tiên cao

training/train_cckd.py:146–178 đọc checkpoint.json và bỏ qua seed đã hoàn tất mà không lưu/đối chiếu cấu hình. Chạy lại cùng output_dir nhưng đổi dataset, split_seed, window_s, backbone, epochs, lambda, dữ liệu hoặc thứ tự phương pháp có thể dùng kết quả cũ và ghi JSON mang cấu hình mới; thêm mode/student có thể gây KeyError.

Cần fingerprint cấu hình thí nghiệm và nguồn dữ liệu, kiểm tra trước khi resume; từ chối checkpoint không khớp. Việc ghi file bằng rename nguyên tử hiện tại là đúng nhưng chỉ bảo đảm file không dở dang, không bảo đảm đúng thí nghiệm. Chưa có bằng chứng rằng các bảng hiện tại thực sự bị lỗi này.

## 3. Cache Module 1 không nhận biết thay đổi cửa sổ/nguồn dữ liệu — ưu tiên cao

training/filterbank.py:53–79 chỉ dùng dataset, shape, sfreq, bands và channels làm khóa. Thay t_start hoặc data_root nhưng giữ shape sẽ đọc lại dữ liệu đã lọc cũ. Kiểm tra bằng hai mảng ngẫu nhiên khác nhau, cùng shape và cache_dir, xác nhận lần thứ hai trả đúng mảng cache của lần đầu.

Cần thêm nguồn dữ liệu, thời điểm bắt đầu, cấu hình bộ lọc và phiên bản tiền xử lý vào khóa; có thể dùng metadata file hoặc hash dữ liệu tùy chi phí. Lỗi này nằm trong pipeline Module 1; Module 2 hiện không dùng cache này.

## 4. Thông số filterbank trong YAML bị bỏ qua ở Module 1 — ưu tiên trung bình

train_teacher.py:266 chỉ truyền sfreq, bands và channels vào load_or_create_cache. Hàm cache không nhận order, ripple_db hoặc normalize_per_trial_channel nên apply_filterbank luôn dùng mặc định order=4, ripple=1, normalize=True. Đổi các trường tương ứng trong YAML không thay đổi tiền xử lý thực tế. Cần truyền chúng xuyên suốt và đưa vào khóa cache. Với YAML hiện tại, mặc định trùng cấu hình nên chưa làm sai các lần chạy đó.

## 5. Kết quả phụ thuộc thứ tự mode/student; CE và KD dùng độ chính xác số khác nhau — vấn đề thiết kế thí nghiệm

train_cckd.py:181 chỉ seed một lần trước teacher, rồi huấn luyện baseline/KD nối tiếp. Số lần huấn luyện, số epoch của mô hình trước và thứ tự --modes/--students thay đổi trạng thái RNG của mô hình sau. Vì vậy cùng nhãn seed chưa bảo đảm cùng initialization hoặc shuffle giữa phương pháp; baseline 1C có thể thay đổi khi chỉ thêm mode cho 3C. cka_analysis.py dùng cùng kiểu này.

Nên tách seed theo teacher và từng cấu hình student; reset RNG trước mỗi phương pháp để baseline và KD bắt đầu từ cùng trạng thái student, dùng generator DataLoader riêng. Đây không tự động vô hiệu hóa phép Wilcoxon ghép theo subject, nhưng làm giảm khả năng kiểm soát yếu tố ngẫu nhiên và độ tái lập của ablation.

Trên CUDA, train_ce dùng autocast + GradScaler (cckd_pipeline.py:106–117), train_kd dùng FP32. Nên thống nhất chế độ số học hoặc công bố khác biệt này. Nếu dùng AMP cho KD, nên giữ phép nhân Gram/normalization ở FP32 để hạn chế overflow. Chưa chạy phép so sánh accuracy AMP/FP32.

## 6. std_acc bị mô tả sai — lỗi báo cáo

cckd_pipeline.py:179–193 lấy trung bình seed cho từng subject, rồi tính std qua các subject. train_cckd.py:222 lại in “mean +/- std over seeds”. Ví dụ hai seed có kết quả giống hệt {subject1:0.2, subject2:0.8} trả std_acc=0.3 dù SD qua seed bằng 0.

Nên đổi nhãn thành SD across subjects after seed averaging và bổ sung trường SD qua seed nếu cần. Accuracy trung bình không bị sai vì lỗi nhãn này.

## 7. Loader MATLAB v7.3 không hỗ trợ struct/group — lỗi tương thích

public_ssvep.py:19–23 chỉ đọc đối tượng HDF5 có thuộc tính shape; group như data/EEG bị bỏ qua. Kiểm tra trực tiếp nhánh fallback với group data chứa EEG cho dictionary rỗng và lỗi “No 4-D EEG tensor found”. Cần duyệt group và xử lý MATLAB object references. Đây là kiểm tra layout HDF5 và nhánh fallback, không phải xác nhận lỗi trên file MAT gốc đang dùng. Loader scipy cho struct BETA và loader v7.3 là hai đường riêng.

## 8. Evaluation chuyển cả tập lên GPU trước khi chia batch — nguy cơ OOM

cckd_pipeline.py:91 và cka_analysis.py:71 gọi torch.from_numpy(X).to(device) trước vòng lặp batch. Vì vậy batch=512 không giới hạn bộ nhớ của toàn tập test/validation trên GPU. Với cửa sổ dài hoặc GPU nhỏ có thể hết VRAM. Nên giữ X trên CPU và chỉ chuyển lát cắt của batch vào GPU. Chưa tái hiện OOM trên phần cứng thực tế.

## Các lưu ý khi diễn giải bài báo

- fixed_split tạo train/val/test không trùng nhau trong từng split. Tuy nhiên các split dùng lại cùng bộ subject: số subject test trùng giữa các cặp split Benchmark liên tiếp (2026/3026/4026/5026/6026) là 2, 3, 4, 3. Đây không phải rò rỉ trong từng split. Cần làm rõ p_t là kiểm định trên các lần chia ngẫu nhiên của cùng cohort, không phải năm cohort độc lập; không suy rộng quá mức ra tổng thể subject. Độ phù hợp của suy luận thống kê cần xét mục tiêu là biến thiên do chia dữ liệu hay tổng quát sang người mới.
- Phần abstract nói chiều cải thiện dương qua cả 5 split ở cả bốn cấu hình, nhưng bảng multisplit có BETA-1C/split 5026 = -0.0625pp. Nên viết “trung bình dương ở cả bốn cấu hình; 19/20 tổ hợp split–cấu hình dương”.
- Pi benchmark đo filterbank và forward riêng rồi cộng trung bình (deploy/pi4_latency/cckd_latency_pi.py:174–187), chưa đo một vòng raw EEG → filterbank → tensor → forward liên tục; torch.from_numpy nằm ngoài phần timing. Đây là ước lượng tổng hai thành phần, nên sửa nhãn hoặc thêm phép đo end-to-end thực.
- YAML/README và smoke test là Module 1 (4 band Chebyshev, chuẩn hóa, LOSO, BN riêng mỗi band). Bài báo và Module 2 là 3 band Butterworth, không chuẩn hóa, fixed split, BN dùng chung. Hai pipeline khác nhau có chủ đích; không được dùng smoke test Module 1 như bằng chứng đã tái lập các thí nghiệm bài báo.
- Khi dò lambda/tau, JSON cuối hiện chỉ lưu test summary, không lưu validation accuracy cho từng cấu hình. Chưa đủ bằng chứng để kết luận siêu tham số có được chọn bằng validation hay test. Nên lưu validation metrics và ghi rõ quy tắc lựa chọn trong bài báo; nếu chọn bằng test phải làm lại phần lựa chọn và đánh giá.

## Kiểm tra đã hoàn tất

- tests/smoke_test.py: PASSED với Python môi trường erd_tta_bci, PyTorch 2.7.1+cu118.
- Forward + backward trên CPU cho cả FB-tCNN, EEGNet, DeepConvNet, CompactCNN, ShallowConvNet: PASSED, logits (2,40), feature dimensions đúng.
- Công thức Gram, stale cache, std và thiếu hỗ trợ HDF5 group: tái hiện bằng ví dụ nhỏ.
- Accuracy bảng overall khớp JSON tier0_multisplit seed-split 2026: Benchmark teacher/3C baseline/3C KD/1C baseline/1C KD = 65.69/50.99/51.44/44.55/45.14%; BETA = 53.91/38.41/39.46/34.62/35.68%.
- Tính lại delta và t-test từ JSON 5 split: Benchmark p = 0.00499435, 0.01033231; BETA p = 0.00022718, 0.02926087. Khớp các giá trị làm tròn trong bài báo. Điều này xác nhận phép tính từ artifact, không xác nhận toàn bộ thiết kế suy luận thống kê hoặc nguồn gốc mọi lần chạy.
- Teacher ở eval và requires_grad=False khi KD; minibatch student/teacher ghép cùng trial; best checkpoint chọn bằng validation, test không tham gia chọn epoch trong pipeline đã đọc.

Ưu tiên xử lý: khớp công thức bài báo/code; chặn resume sai cấu hình; sửa cache và truyền cấu hình bộ lọc; rồi chuẩn hóa RNG/precision và sửa nhãn thống kê. Chưa sửa implementation để tránh thay đổi mục tiêu thí nghiệm và kết quả đã có khi nhiệm vụ hiện tại là rà soát.
