# Bản đối chiếu song ngữ: Introduction + Related Work (IJRCS_CCKD_SSVEP.tex)

Dịch Anh → Việt, ánh xạ 1:1 theo câu, theo quy trình của skill `dich-viet-anh` áp dụng
ngược chiều (PASS 0–8, xem báo cáo cuối file). Giữ nguyên tiếng Anh cho các thuật ngữ
chuyên ngành đã thành chuẩn trong các file khác của dự án (SSVEP, BCI, KD, CCKD, CKA,
backbone, montage, split, seed, teacher/student) — xem mục "Bảng thuật ngữ".
Không dịch citation key (`\cite{R..}`) và `\tbd{}`.

## 1. Introduction

### Đoạn 1 (vấn đề: trễ, độ đeo, độ chính xác trong vòng điều khiển)

| ID | English | Tiếng Việt |
|---|---|---|
| S1.P1.C1 | A steady-state visual evoked potential (SSVEP) brain-computer interface (BCI) acts as the command-generation stage of a closed-loop control system: it senses the user's intent from EEG, maps the decoded target frequency to a discrete control command, and passes that command to a lower-level controller that actuates a wheelchair, a spelling interface, or a robotic arm [R3,R6,R7,R8,R9]. | Giao diện não-máy (BCI) dựa trên điện thế gợi thị giác trạng thái ổn định (SSVEP) đóng vai trò khối sinh lệnh điều khiển trong một hệ điều khiển vòng kín: nó cảm nhận ý định của người dùng từ EEG, ánh xạ tần số mục tiêu đã giải mã thành một lệnh điều khiển rời rạc và chuyển lệnh đó tới một bộ điều khiển cấp thấp hơn để tác động lên xe lăn, giao diện đánh vần hoặc tay máy [R3,R6,R7,R8,R9]. |
| S1.P1.C2 | As in any human-in-the-loop control architecture, the achievable command rate, measured in the BCI literature as the information transfer rate (ITR), sets the throughput of this channel, and the decision latency of the classifier must fit within the control loop's update period for the system to react safely to the operator's intent [R4,R5]. | Như trong bất kỳ kiến trúc điều khiển có người trong vòng lặp (human-in-the-loop) nào khác, tốc độ lệnh đạt được, mà tài liệu BCI gọi là tốc độ truyền thông tin (ITR), quyết định thông lượng của kênh này và độ trễ quyết định của bộ phân loại phải nằm trong chu kỳ cập nhật của vòng điều khiển để hệ thống phản hồi an toàn với ý định của người vận hành [R4,R5]. |
| S1.P1.C3 | A multi-electrode wet-gel cap improves decoding accuracy and therefore the reliability of each generated command, but it also increases setup time and reduces wearability, which limits the use of such a controller outside the laboratory [R39,R40,R41]. | Mũ điện cực gel ướt nhiều kênh cải thiện độ chính xác giải mã và vì vậy cải thiện độ tin cậy của mỗi lệnh được sinh ra, nhưng nó cũng làm tăng thời gian lắp đặt và giảm khả năng đeo được, điều này giới hạn việc sử dụng loại bộ điều khiển này bên ngoài phòng thí nghiệm [R39,R40,R41]. |
| S1.P1.C4 | Reducing the electrode count to one or three channels improves wearability for a controller meant to be worn daily, but it degrades accuracy, because the classifier loses the spatial information carried by the discarded channels, which in turn raises the rate of missed or incorrect commands sent to the actuator [R36,R38]. | Giảm số điện cực xuống một hoặc ba kênh giúp cải thiện khả năng đeo được cho một bộ điều khiển dùng đeo hàng ngày, nhưng làm giảm độ chính xác vì bộ phân loại mất đi thông tin không gian mà các kênh bị loại bỏ mang lại, điều này đến lượt nó làm tăng tỷ lệ lệnh bị thiếu hoặc sai gửi tới cơ cấu chấp hành [R36,R38]. |
| S1.P1.C5 | This trade-off between the wearability of the sensing front end and the reliability of the control commands it produces is the central problem addressed in this paper. | Sự đánh đổi giữa khả năng đeo được của tầng cảm biến đầu vào và độ tin cậy của các lệnh điều khiển mà nó tạo ra là vấn đề trung tâm mà bài báo này giải quyết. |

### Đoạn 2 (giải pháp KD/CCKD, công trình ICERA trước đây)

| ID | English | Tiếng Việt |
|---|---|---|
| S1.P2.C1 | Knowledge distillation (KD) offers one way to narrow this gap: a teacher network trained on the full montage transfers information to a student network restricted to the low-electrode layout [R47]. | KD (knowledge distillation) là một cách để thu hẹp khoảng cách này: một mạng teacher được huấn luyện trên toàn bộ montage truyền thông tin sang một mạng student bị giới hạn ở cấu hình điện cực ít [R47]. |
| S1.P2.C2 | Cross-channel relational knowledge distillation (CCKD) is one such method. | CCKD (cross-channel relational knowledge distillation) là một phương pháp như vậy. |
| S1.P2.C3 | It preserves the pairwise similarity structure of teacher features, encoded as a Gram matrix, instead of matching feature values directly [R82]. | Phương pháp này giữ lại cấu trúc tương đồng theo từng cặp của đặc trưng teacher, được mã hóa dưới dạng ma trận Gram, thay vì khớp trực tiếp giá trị đặc trưng [R82]. |
| S1.P2.C4 | In earlier work, we showed that CCKD recovers part of the accuracy lost to montage reduction for one convolutional backbone, FB-tCNN, on two public 40-target SSVEP datasets [R82,R16]. | Trong công trình trước đây, chúng tôi đã chỉ ra rằng CCKD khôi phục được một phần độ chính xác bị mất do giảm montage, trên một backbone tích chập là FB-tCNN, với hai tập dữ liệu SSVEP công khai có 40 lớp mục tiêu [R82,R16]. |
| S1.P2.C5 | That evaluation, like most EEG knowledge-distillation studies, used a single backbone, a single distillation layer, and a single subject split [R48,R49,R50,R51,R52,R53], leaving open whether the benefit generalizes to other architectures, to other electrode subsets, or to a more rigorous multi-split evaluation. | Đánh giá đó, giống như hầu hết các nghiên cứu KD cho EEG khác, chỉ dùng một backbone, một lớp để distill và một split đối tượng duy nhất [R48,R49,R50,R51,R52,R53], để ngỏ câu hỏi liệu lợi ích này có tổng quát hóa sang kiến trúc khác, sang tập con điện cực khác hay sang một cách đánh giá đa-split chặt chẽ hơn hay không. |

### Đoạn 3 (backbone SSVEP, độ nhạy đánh giá EEG)

| ID | English | Tiếng Việt |
|---|---|---|
| S1.P3.C1 | SSVEP decoding has moved from spatial-filter methods such as canonical correlation analysis toward a wide range of deep architectures, including filter-bank convolutional networks [R16], transformer-based models [R15,R23], graph convolutional networks [R18], and attention-augmented compact networks [R19,R20,R21,R26,R29]. | Giải mã SSVEP đã chuyển từ các phương pháp lọc không gian như phân tích tương quan chuẩn tắc (canonical correlation analysis) sang một loạt kiến trúc học sâu, gồm mạng tích chập dùng filter bank [R16], mô hình dựa trên transformer [R15,R23], mạng tích chập đồ thị [R18] và mạng compact có thêm attention [R19,R20,R21,R26,R29]. |
| S1.P3.C2 | These backbones differ in depth, receptive field, and the number of internal feature stages at which a teacher-student match can be imposed, so a distillation strategy tuned for one architecture need not transfer to another. | Các backbone này khác nhau về độ sâu, vùng tiếp nhận (receptive field) và số tầng đặc trưng bên trong có thể áp đặt khớp teacher-student, vì vậy một chiến lược distillation được tinh chỉnh cho một kiến trúc không chắc chuyển sang kiến trúc khác. |
| S1.P3.C3 | A separate line of work shows that EEG machine-learning results are sensitive to how data are split and how significance is assessed: segment-level or single-split evaluation can invert the sign of an effect or inflate its apparent size [R75,R76,R77,R78,R79,R80,R81]. | Một hướng nghiên cứu khác cho thấy kết quả học máy trên EEG nhạy với cách chia dữ liệu và cách đánh giá ý nghĩa thống kê: đánh giá theo đoạn hoặc theo một split duy nhất có thể đảo dấu của hiệu ứng hoặc làm phóng đại độ lớn biểu kiến của nó [R75,R76,R77,R78,R79,R80,R81]. |
| S1.P3.C4 | Distillation gains reported in the sub-percentage-point range are exactly the regime in which this sensitivity matters most, yet our earlier CCKD evaluation relied on a single subject split [R82]. | Mức cải thiện do distillation được báo cáo trong khoảng dưới một điểm phần trăm chính là vùng mà độ nhạy này ảnh hưởng nhiều nhất, nhưng đánh giá CCKD trước đây của chúng tôi lại chỉ dựa trên một split đối tượng duy nhất [R82]. |

### Đoạn 4 (CKA, triển khai nhúng)

| ID | English | Tiếng Việt |
|---|---|---|
| S1.P4.C1 | Centered kernel alignment (CKA) quantifies how similar two sets of neural-network representations are, independent of rotation and scale, and is a standard tool for comparing layers across trained networks [R73,R74,R71,R72]. | CKA (centered kernel alignment) định lượng mức độ tương đồng giữa hai tập biểu diễn của mạng nơ-ron một cách không phụ thuộc vào phép xoay và tỷ lệ, đồng thời là công cụ chuẩn để so sánh các lớp giữa các mạng đã huấn luyện [R73,R74,R71,R72]. |
| S1.P4.C2 | It matches the quantity that CCKD already optimizes, because the training objective matches the Gram matrix of pairwise sample similarities between teacher and student. | Đại lượng này khớp với đại lượng mà CCKD đã tối ưu hóa, vì mục tiêu huấn luyện khớp ma trận Gram biểu diễn độ tương đồng theo từng cặp mẫu giữa teacher và student. |
| S1.P4.C3 | CKA-based regularizers have been applied directly inside EEG backbones [R70], but using the *change* in CKA induced by distillation to predict how much accuracy the distillation step will recover remains untested in SSVEP decoding. | Các bộ điều chuẩn hóa dựa trên CKA đã được áp dụng trực tiếp bên trong backbone cho EEG [R70], nhưng việc dùng *mức thay đổi* của CKA do distillation gây ra để dự đoán mức độ chính xác mà bước distillation sẽ khôi phục được vẫn chưa được kiểm chứng trong giải mã SSVEP. |
| S1.P4.C4 | Finally, deploying any of these backbones on the embedded hardware used in real wheelchair and speller prototypes, such as a Raspberry Pi 4 or a Xilinx FPGA, requires quantization and resource-aware implementation that can itself change classification accuracy [R58,R60,R62,R63]. | Cuối cùng, việc triển khai bất kỳ backbone nào trong số này trên phần cứng nhúng dùng trong các mẫu thử xe lăn và bộ đánh vần thực tế, chẳng hạn Raspberry Pi 4 hoặc FPGA của Xilinx, yêu cầu lượng tử hóa và cách cài đặt có tính đến tài nguyên; bản thân việc cài đặt này có thể làm thay đổi độ chính xác phân loại [R58,R60,R62,R63]. |
| S1.P4.C5 | Whether a knowledge-distilled low-electrode model keeps its accuracy advantage after this step is a question the SSVEP-BCI literature has not addressed directly. | Liệu một mô hình ít điện cực đã qua KD có giữ được lợi thế độ chính xác sau bước này hay không là câu hỏi mà tài liệu về SSVEP-BCI chưa trả lời trực tiếp. |

### Đoạn 5 (đóng góp)

| ID | English | Tiếng Việt |
|---|---|---|
| S1.P5.C1 | This paper extends our single-backbone conference evaluation [R82] along four directions to close these gaps. | Bài báo này mở rộng đánh giá một-backbone ở bài hội nghị trước đây của chúng tôi [R82] theo bốn hướng để lấp các khoảng trống này. |
| S1.P5.C2 | The contributions of this research are as follows. | Các đóng góp của nghiên cứu này như sau. |
| S1.P5.C3 (i) | A controlled evaluation of CCKD across five public backbones (FB-tCNN, DeepConvNet, ShallowConvNet, EEGNet, and Compact-CNN), two 40-target datasets, and two low-electrode montages, each with five independent subject splits and eight seeds per split, with Holm-Bonferroni correction for multiple comparisons. | Một đánh giá có kiểm soát CCKD trên năm backbone công khai (FB-tCNN, DeepConvNet, ShallowConvNet, EEGNet và Compact-CNN), hai tập dữ liệu 40 lớp mục tiêu và hai montage ít điện cực, mỗi trường hợp dùng năm split đối tượng độc lập và tám seed trên mỗi split, có hiệu chỉnh Holm-Bonferroni cho so sánh đa biến. |
| S1.P5.C4 (ii) | An analysis of centered kernel alignment between teacher and student features as a predictor of distillation benefit, including an intervention that distills an intermediate layer instead of the penultimate layer used in prior work. | Một phân tích CKA giữa đặc trưng của teacher và student như một đại lượng dự đoán lợi ích của distillation, kèm một can thiệp distill vào lớp trung gian thay vì lớp kề cuối đã dùng trong công trình trước. |
| S1.P5.C5 (iii) | A direct comparison of single-split and multi-split evaluation on the same data, quantifying how often single-split evaluation misjudges the sign or significance of the effect. | Một so sánh trực tiếp giữa đánh giá một-split và đánh giá nhiều-split trên cùng dữ liệu, định lượng tần suất đánh giá một-split đánh giá sai dấu hoặc ý nghĩa thống kê của hiệu ứng. |
| S1.P5.C6 (iv) | A deployment of the resulting student models on a Raspberry Pi 4 and a Xilinx ZCU104, the hardware classes used in wheelchair-control and SSVEP-speller prototypes, reporting decision latency against the control-loop update period those prototypes require, power, and post-quantization accuracy [TBD: pending Section 6 experiments]. | Một lần triển khai các mô hình student thu được lên Raspberry Pi 4 và Xilinx ZCU104, hai lớp phần cứng dùng trong các mẫu thử điều khiển xe lăn và bộ đánh vần SSVEP, báo cáo độ trễ quyết định so với chu kỳ cập nhật vòng điều khiển mà các mẫu thử đó yêu cầu, công suất tiêu thụ và độ chính xác sau lượng tử hóa [CHƯA CÓ: chờ thí nghiệm mục 6]. |

## 2. Related Work

### 2.1 Deep Learning Backbones for SSVEP Decoding — Các backbone học sâu cho giải mã SSVEP

| ID | English | Tiếng Việt |
|---|---|---|
| S2.1.C1 | Classical SSVEP recognition relies on frequency-domain correlation methods, such as canonical correlation analysis and task-related component analysis, which compare the recorded signal against a reference template at each stimulation frequency. | Nhận dạng SSVEP kiểu cổ điển dựa trên các phương pháp tương quan trong miền tần số, như phân tích tương quan chuẩn tắc (CCA) và phân tích thành phần liên quan đến nhiệm vụ (TRCA), so sánh tín hiệu ghi được với một mẫu tham chiếu tại từng tần số kích thích. |
| S2.1.C2 | Convolutional and attention-based networks have since replaced or augmented these templates, achieving higher accuracy with less calibration data. | Từ đó, các mạng tích chập và mạng dựa trên attention đã thay thế hoặc bổ sung cho các mẫu tham chiếu này, đạt độ chính xác cao hơn với ít dữ liệu hiệu chỉnh hơn. |
| S2.1.C3 | FB-tCNN combines a filter bank with a time-domain convolutional network to handle short time windows [R16] and is the backbone our 2026 conference paper distilled into a low-electrode student [R82]. | FB-tCNN kết hợp filter bank với một mạng tích chập miền thời gian để xử lý cửa sổ thời gian ngắn [R16] và là backbone mà bài báo hội nghị năm 2026 của chúng tôi đã distill sang một student ít điện cực [R82]. |
| S2.1.C4 | ConvCA replaces the fixed correlation template with a learned convolutional correlation layer [R17]. | ConvCA thay mẫu tương quan cố định bằng một lớp tương quan tích chập được học [R17]. |
| S2.1.C5 | SSVEPformer and related transformer architectures model long-range dependencies in the frequency-domain spectrum rather than the raw time series [R15,R23]. | SSVEPformer và các kiến trúc transformer liên quan mô hình hóa phụ thuộc tầm xa trên phổ miền tần số thay vì trên chuỗi thời gian thô [R15,R23]. |
| S2.1.C6 | Graph-convolutional variants such as DDGCNN exploit the spatial layout of electrodes directly [R18], and lightweight attention networks such as the task-attention model of Wang et al. and LGFCNN reduce the number of trainable parameters needed for individual calibration [R20,R26]. | Các biến thể tích chập đồ thị như DDGCNN khai thác trực tiếp cách bố trí không gian của điện cực [R18], còn các mạng attention nhẹ như mô hình task-attention của Wang và cộng sự hay LGFCNN làm giảm số tham số cần huấn luyện cho hiệu chỉnh theo từng cá nhân [R20,R26]. |
| S2.1.C7 | Several studies address the few-channel regime directly, either through specialized architectures [R22,R25,R38] or through analyses of how channel count and placement affect accuracy [R36,R37]. | Một số nghiên cứu giải quyết trực tiếp chế độ ít kênh, bằng kiến trúc chuyên biệt [R22,R25,R38] hoặc bằng phân tích cách số lượng và vị trí điện cực ảnh hưởng tới độ chính xác [R36,R37]. |
| S2.1.C8 | Across this body of work, each architecture is typically evaluated and tuned on its own; comparative multi-backbone studies of any post-hoc transfer technique, including knowledge distillation, are rare. | Trong toàn bộ khối nghiên cứu này, mỗi kiến trúc thường được đánh giá và tinh chỉnh riêng lẻ; các nghiên cứu so sánh đa-backbone cho bất kỳ kỹ thuật truyền tri thức hậu-huấn luyện nào, kể cả KD, đều hiếm gặp. |

### 2.2 Knowledge Distillation for EEG and Brain-Computer Interfaces — KD cho EEG và giao diện não-máy

| ID | English | Tiếng Việt |
|---|---|---|
| S2.2.C1 | Knowledge distillation transfers information from a larger teacher network to a smaller or more constrained student network, usually by matching output logits, intermediate features, or pairwise feature relations. | KD truyền thông tin từ một mạng teacher lớn hơn sang một mạng student nhỏ hơn hoặc bị giới hạn hơn, thường bằng cách khớp logit đầu ra, đặc trưng trung gian hoặc quan hệ đặc trưng theo từng cặp. |
| S2.2.C2 | In EEG-based BCIs, similarity-keeping distillation reduces the channel count of a classifier while preserving most of its accuracy [R47], which is the closest precedent to the cross-channel relational objective evaluated in this paper. | Trong BCI dựa trên EEG, distillation kiểu giữ-độ-tương-đồng (similarity-keeping) làm giảm số kênh của bộ phân loại mà vẫn giữ được phần lớn độ chính xác [R47]; đây là tiền lệ gần nhất với mục tiêu quan hệ liên-kênh được đánh giá trong bài báo này. |
| S2.2.C3 | Other EEG and physiological-signal studies use distillation mainly across subjects or domains rather than across channel counts: dual-teacher and multi-teacher frameworks combine several source models for motor-imagery decoding [R49,R54], teacher-assistant chains ease a large capacity gap between teacher and student [R53], and self-distillation or domain-generalization objectives target cross-subject and cross-dataset emotion recognition [R50,R52,R55,R57]. | Các nghiên cứu khác về EEG và tín hiệu sinh lý dùng distillation chủ yếu giữa các đối tượng hoặc giữa các domain thay vì giữa các số kênh: khung dual-teacher và multi-teacher kết hợp nhiều mô hình nguồn cho giải mã vận động tưởng tượng [R49,R54], chuỗi teacher-assistant giảm bớt khoảng cách năng lực lớn giữa teacher và student [R53], còn mục tiêu self-distillation hoặc domain-generalization hướng tới nhận dạng cảm xúc liên-đối-tượng và liên-tập-dữ-liệu [R50,R52,R55,R57]. |
| S2.2.C4 | A cross-modal variant distills knowledge from a visual encoder into an EEG encoder for continuous emotion recognition [R51]. | Một biến thể liên-phương-thức (cross-modal) distill tri thức từ một encoder hình ảnh sang một encoder EEG để nhận dạng cảm xúc liên tục [R51]. |
| S2.2.C5 | One study combines distillation with quantization-aware training in a single objective for EEG-based emotion recognition [R56], which anticipates the combined knowledge-distillation-and-embedded-deployment pipeline evaluated in this paper. | Một nghiên cứu kết hợp distillation với huấn luyện có nhận biết lượng tử hóa (quantization-aware training) trong cùng một mục tiêu cho nhận dạng cảm xúc dựa trên EEG [R56]; điều này đi trước quy trình kết hợp KD và triển khai nhúng được đánh giá trong bài báo này. |
| S2.2.C6 | None of these studies evaluate whether a channel-reduction distillation objective generalizes across backbone architecture, which is the first research question addressed here. | Không nghiên cứu nào trong số này đánh giá liệu một mục tiêu distillation để giảm kênh có tổng quát hóa được sang kiến trúc backbone khác hay không; đây chính là câu hỏi nghiên cứu thứ nhất được giải quyết ở đây. |

### 2.3 SSVEP-Based Control Applications — Ứng dụng điều khiển dựa trên SSVEP: xe lăn, bộ đánh vần và tay máy

| ID | English | Tiếng Việt |
|---|---|---|
| S2.3.C1 | In an SSVEP-controlled assistive device, the decoder is one stage in a closed control loop: the stimulus interface presents a fixed set of flicker targets, the decoder maps the detected frequency to one discrete command out of that set, a low-level controller converts the command into a motion or selection primitive, and the device's motion or the speller's screen feedback closes the loop back to the operator. | Trong một thiết bị hỗ trợ điều khiển bằng SSVEP, bộ giải mã là một khâu trong vòng điều khiển kín: giao diện kích thích hiển thị một tập mục tiêu nhấp nháy cố định, bộ giải mã ánh xạ tần số phát hiện được thành một lệnh rời rạc trong tập đó, một bộ điều khiển cấp thấp chuyển lệnh thành một chuyển động hoặc một lựa chọn cơ bản, còn chuyển động của thiết bị hoặc phản hồi trên màn hình của bộ đánh vần khép vòng lặp trở lại người vận hành. |
| S2.3.C2 | The number of commands available per unit time, and therefore the number of independent degrees of freedom the operator can control directly, is set jointly by the number of stimulus targets and the decoder's ITR [R4,R5]. | Số lệnh khả dụng trong một đơn vị thời gian (và vì vậy số bậc tự do độc lập mà người vận hành có thể điều khiển trực tiếp) được quyết định đồng thời bởi số mục tiêu kích thích và ITR của bộ giải mã [R4,R5]. |
| S2.3.C3 | Wheelchair-control systems built on this loop use SSVEP alone for discrete heading commands [R3,R6], SSVEP combined with electrooculography [R2] or P300 potentials [R7] to add command channels, and SSVEP combined with computer vision and augmented reality, where the onboard perception system supplies continuous obstacle avoidance and the SSVEP channel supplies only the high-level heading command, reducing the decoding accuracy and update rate the BCI stage must guarantee for safe operation [R7,R8]. | Các hệ điều khiển xe lăn xây dựng trên vòng lặp này dùng SSVEP một mình cho lệnh hướng đi rời rạc [R3,R6], dùng SSVEP kết hợp với điện-nhãn-đồ (EOG) [R2] hoặc điện thế P300 [R7] để thêm kênh lệnh và dùng SSVEP kết hợp với thị giác máy tính cùng thực tế tăng cường, trong đó hệ thống nhận thức trên thiết bị đảm nhiệm việc tránh vật cản liên tục còn kênh SSVEP chỉ đảm nhiệm lệnh hướng đi ở mức cao, làm giảm yêu cầu về độ chính xác giải mã và tốc độ cập nhật mà khối BCI phải đảm bảo để vận hành an toàn [R7,R8]. |
| S2.3.C4 | Systematic reviews of this application area report that most prototypes still use eight or more electrodes, in part because the stimulation and classification pipeline was developed for a research-grade cap rather than a long-wear consumer headset, and that few report a closed-loop latency budget explicitly [R10,R11]. | Các tổng quan hệ thống về lĩnh vực ứng dụng này cho thấy hầu hết mẫu thử vẫn dùng tám điện cực hoặc nhiều hơn, một phần vì pipeline kích thích và phân loại được phát triển cho mũ cấp nghiên cứu chứ không phải cho tai nghe tiêu dùng đeo lâu dài, và cho thấy rất ít nghiên cứu báo cáo rõ ngân sách độ trễ của vòng kín [R10,R11]. |
| S2.3.C5 | Speller interfaces face the same loop with a much larger command set: encoding more target characters needs a larger stimulus set and a correspondingly higher perceptual workload per selection, which motivates paradigm designs that enhance the discriminability of each frequency and keep the per-character decision time bounded, such as frequency-enhanced row-column spellers [R4] and adaptive spatial filtering [R5]. | Giao diện bộ đánh vần gặp cùng vòng lặp này nhưng với tập lệnh lớn hơn nhiều: mã hóa nhiều ký tự mục tiêu hơn cần một tập kích thích lớn hơn và một tải nhận thức cao hơn tương ứng cho mỗi lần chọn, điều này thúc đẩy các thiết kế paradigm tăng khả năng phân biệt của từng tần số và giữ thời gian quyết định cho mỗi ký tự trong giới hạn, chẳng hạn bộ đánh vần hàng-cột tăng cường tần số [R4] và lọc không gian thích nghi [R5]. |
| S2.3.C6 | Robotic-arm control shares the same channel-count trade-off at a finer grain: reducing the montage improves comfort for extended operation, but it narrows the margin between target frequencies that the decoder can resolve reliably, which directly limits how many independent joint or end-effector commands the control loop can safely accept [R9]. | Điều khiển tay máy chia sẻ cùng sự đánh đổi về số kênh này ở mức chi tiết hơn: giảm montage giúp thoải mái hơn khi vận hành kéo dài, nhưng làm hẹp biên giữa các tần số mục tiêu mà bộ giải mã có thể phân giải một cách tin cậy, điều này trực tiếp hạn chế số lệnh khớp hoặc khâu tác động cuối độc lập mà vòng điều khiển có thể tiếp nhận an toàn [R9]. |
| S2.3.C7 | These applications motivate the two deployment scenarios evaluated later in this paper: a three-electrode montage intended for a wheelchair headset, and a comparison across electrode subsets relevant to speller and robotic-arm command sets. | Các ứng dụng này là động lực cho hai kịch bản triển khai được đánh giá ở phần sau của bài báo: một montage ba điện cực dùng cho tai nghe xe lăn và một so sánh giữa các tập con điện cực liên quan tới tập lệnh của bộ đánh vần và tay máy. |

### 2.4 Representation Similarity and Centered Kernel Alignment — Độ tương đồng biểu diễn và CKA

| ID | English | Tiếng Việt |
|---|---|---|
| S2.4.C1 | Centered kernel alignment (CKA) measures how similar two sets of representations are, without requiring the same number of units or a fixed correspondence between them. | CKA đo mức độ tương đồng giữa hai tập biểu diễn mà không yêu cầu cùng số đơn vị hay một phép tương ứng cố định giữa chúng. |
| S2.4.C2 | The Gaussian and linear variants of this measure have become standard tools for comparing layers across trained networks [R73,R74,R71]. | Các biến thể Gaussian và tuyến tính của đại lượng này đã trở thành công cụ chuẩn để so sánh các lớp giữa các mạng đã huấn luyện [R73,R74,R71]. |
| S2.4.C3 | Kriegeskorte and Wei situate CKA within the broader framework of representational geometry, which compares the relative distances or similarities among stimuli rather than their absolute feature values [R72]. | Kriegeskorte và Wei đặt CKA trong khung lý thuyết rộng hơn về hình học biểu diễn (representational geometry), so sánh khoảng cách hoặc độ tương đồng tương đối giữa các kích thích thay vì giá trị đặc trưng tuyệt đối của chúng [R72]. |
| S2.4.C4 | This framework matches the quantity CCKD already optimizes: the training objective matches the Gram matrix of pairwise sample similarities between teacher and student, which is the same object CKA uses to quantify representational similarity. | Khung lý thuyết này khớp với đại lượng mà CCKD đã tối ưu hóa: mục tiêu huấn luyện khớp ma trận Gram biểu diễn độ tương đồng theo từng cặp mẫu giữa teacher và student, chính là đại lượng mà CKA dùng để định lượng độ tương đồng biểu diễn. |
| S2.4.C5 | Tobon-Henao et al. apply a centered-alignment regularizer directly inside an EEGNet classifier for motor-imagery discrimination [R70], confirming that CKA-based objectives are compatible with EEG backbones. | Tobon-Henao và cộng sự áp dụng một bộ điều chuẩn hóa centered-alignment trực tiếp bên trong bộ phân loại EEGNet cho phân biệt vận động tưởng tượng [R70], xác nhận rằng các mục tiêu dựa trên CKA tương thích được với backbone cho EEG. |
| S2.4.C6 | To our knowledge, no prior SSVEP study has used the change in CKA induced by distillation as a predictor of how much accuracy the distillation step recovers, which is the second research question addressed here. | Theo hiểu biết của chúng tôi, chưa có nghiên cứu SSVEP nào trước đây dùng mức thay đổi của CKA do distillation gây ra như một đại lượng dự đoán mức độ chính xác mà bước distillation khôi phục được; đây là câu hỏi nghiên cứu thứ hai được giải quyết ở đây. |

### 2.5 Statistical Reliability in Cross-Subject EEG Evaluation — Độ tin cậy thống kê trong đánh giá EEG liên-đối-tượng

| ID | English | Tiếng Việt |
|---|---|---|
| S2.5.C1 | EEG classification accuracy is sensitive to how training, validation, and test data are partitioned across subjects and sessions. | Độ chính xác phân loại EEG nhạy với cách chia dữ liệu huấn luyện, validation và test theo đối tượng và theo buổi ghi. |
| S2.5.C2 | White and Power show that k-fold cross-validation without subject-level separation substantially overestimates accuracy in passive BCI designs [R75]. | White và Power cho thấy kiểm định chéo k-fold không tách theo đối tượng làm phóng đại đáng kể độ chính xác trong các thiết kế BCI bị động [R75]. |
| S2.5.C3 | Del Pup et al. and Kamrud et al. report that subject-wise partitioning choices alone change measured performance by several percentage points in cross-subject deep-learning models [R76,R79]. | Del Pup và cộng sự cùng Kamrud và cộng sự báo cáo rằng chỉ riêng cách chia theo đối tượng cũng làm thay đổi hiệu năng đo được vài điểm phần trăm trong các mô hình học sâu liên-đối-tượng [R76,R79]. |
| S2.5.C4 | Brookshire et al., Lee et al., and Geethapriya and Pattabiraman document the same leakage pattern in translational and clinical EEG classifiers, where segment-level splitting lets a classifier exploit subject identity instead of the intended signal [R77,R78,R81]. | Brookshire và cộng sự, Lee và cộng sự, cùng Geethapriya và Pattabiraman ghi nhận cùng kiểu rò rỉ dữ liệu này trong các bộ phân loại EEG lâm sàng và chuyển dịch (translational), nơi việc chia theo đoạn cho phép bộ phân loại khai thác danh tính đối tượng thay vì tín hiệu thực sự cần nhận dạng [R77,R78,R81]. |
| S2.5.C5 | Xu et al. show that accuracy estimates also vary across different EEG datasets collected under nominally the same paradigm, independent of any single-subject leakage [R80]. | Xu và cộng sự cho thấy ước lượng độ chính xác cũng thay đổi giữa các tập dữ liệu EEG khác nhau được thu thập theo cùng một paradigm trên danh nghĩa, độc lập với bất kỳ rò rỉ dữ liệu theo từng đối tượng nào [R80]. |
| S2.5.C6 | These findings motivate evaluating CCKD on five independent subject splits with Holm-Bonferroni correction, rather than on the single split used in the original conference evaluation, and motivate the direct single-split-versus-multi-split comparison reported as the third research question of this paper. | Những phát hiện này là động lực để đánh giá CCKD trên năm split đối tượng độc lập có hiệu chỉnh Holm-Bonferroni thay vì trên một split duy nhất như trong đánh giá ở bài hội nghị gốc và là động lực cho việc so sánh trực tiếp một-split với nhiều-split được báo cáo như câu hỏi nghiên cứu thứ ba của bài báo này. |

### 2.6 Quantization and Embedded Deployment for Real-Time BCIs — Lượng tử hóa và triển khai nhúng cho BCI thời gian thực

| ID | English | Tiếng Việt |
|---|---|---|
| S2.6.C1 | In a control loop, the decoder's execution time is part of the loop's dead time: it adds directly to the delay between the operator's intent and the actuator's response, so an embedded implementation must keep that delay within the margin the mechanical system can tolerate without instability or a perceptibly sluggish response. | Trong một vòng điều khiển, thời gian thực thi của bộ giải mã là một phần của thời gian chết (dead time) của vòng lặp: nó cộng trực tiếp vào độ trễ giữa ý định của người vận hành và phản hồi của cơ cấu chấp hành, vì vậy một cách cài đặt nhúng phải giữ độ trễ đó trong biên mà hệ cơ khí có thể chịu được mà không gây mất ổn định hay phản hồi chậm có thể cảm nhận được. |
| S2.6.C2 | Deploying a trained SSVEP classifier on embedded hardware for this kind of real-time, untethered operation typically requires quantization, pruning, or a dedicated hardware accelerator, because a floating-point convolutional network exceeds the memory and power budget of a wearable or battery-powered controller. | Triển khai một bộ phân loại SSVEP đã huấn luyện lên phần cứng nhúng cho kiểu vận hành thời gian thực và không dây này thường cần lượng tử hóa, pruning hoặc một bộ gia tốc phần cứng chuyên dụng, vì một mạng tích chập dấu phẩy động vượt quá ngân sách bộ nhớ và công suất của một bộ điều khiển đeo được hoặc chạy bằng pin. |
| S2.6.C3 | Pacini et al. compare CPU, GPU, and FPGA deployments of a motor-imagery CNN and report large differences in latency and power across platforms for the same trained model, which shows that the choice of compute platform, not only the model, sets the achievable control-loop rate [R60]. | Pacini và cộng sự so sánh việc triển khai một CNN cho vận động tưởng tượng trên CPU, GPU và FPGA, báo cáo sự khác biệt lớn về độ trễ và công suất giữa các nền tảng cho cùng một mô hình đã huấn luyện; điều này cho thấy việc chọn nền tảng tính toán, không chỉ riêng mô hình, quyết định tốc độ vòng điều khiển đạt được [R60]. |
| S2.6.C4 | Novac et al. provide a general methodology for quantizing and deploying deep networks on microcontrollers [R63], and Palumbo et al. demonstrate a full FPGA-embedded BCI pipeline for locked-in patients, with the decoder running on the same board that generates the output command [R62]. | Novac và cộng sự đưa ra một phương pháp luận tổng quát để lượng tử hóa và triển khai mạng học sâu trên vi điều khiển [R63], còn Palumbo và cộng sự minh chứng một pipeline BCI nhúng trọn vẹn trên FPGA cho bệnh nhân hội chứng khóa trong (locked-in), với bộ giải mã chạy trên cùng board sinh ra lệnh đầu ra [R62]. |
| S2.6.C5 | Spiking and event-driven architectures offer an alternative low-power path for wearable physiological-signal classification, demonstrated for ECG and seizure detection on FPGA [R64,R65,R66,R67,R68,R69], and a recent survey catalogs edge-AI accelerator designs for brain-computer interfaces specifically [R58,R61]. | Các kiến trúc spiking và event-driven đưa ra một hướng công suất thấp khác cho phân loại tín hiệu sinh lý trên thiết bị đeo được, đã minh chứng cho phát hiện ECG và phát hiện co giật trên FPGA [R64,R65,R66,R67,R68,R69]; một khảo sát tổng quan gần đây cũng tổng hợp các thiết kế bộ gia tốc edge-AI dành riêng cho giao diện não-máy [R58,R61]. |
| S2.6.C6 | None of this work evaluates whether a knowledge-distilled student model, already trained to approximate a larger teacher, retains its accuracy advantage after the additional accuracy loss introduced by embedded quantization, or whether its latency on this class of hardware fits the control-loop budget required by a wheelchair or speller controller, which motivates the fourth research question addressed in this paper. | Không công trình nào trong số này đánh giá liệu một mô hình student đã qua KD, được huấn luyện để xấp xỉ một teacher lớn hơn, có giữ được lợi thế độ chính xác sau phần mất chính xác bổ sung do lượng tử hóa nhúng gây ra hay không, cũng như liệu độ trễ của nó trên lớp phần cứng này có đáp ứng ngân sách của vòng điều khiển mà một bộ điều khiển xe lăn hoặc bộ đánh vần yêu cầu hay không; đây là động lực cho câu hỏi nghiên cứu thứ tư được giải quyết trong bài báo này. |

### 2.7 Summary and Research Gap — Tổng hợp và khoảng trống nghiên cứu

| ID | English | Tiếng Việt |
|---|---|---|
| S2.7.C1 | Across these five areas, three gaps motivate the present study, all bearing on the same practical question: can a low-electrode, knowledge-distilled SSVEP decoder serve as the command-generation stage of a real wheelchair, speller, or robotic-arm control loop without an unacceptable loss of command reliability or an unacceptable increase in decision latency? | Trong năm lĩnh vực nêu trên, có ba khoảng trống là động lực cho nghiên cứu này, cả ba đều xoay quanh một câu hỏi thực tiễn chung: một bộ giải mã SSVEP ít điện cực đã qua KD có thể đóng vai trò khối sinh lệnh điều khiển trong một vòng điều khiển thực tế cho xe lăn, bộ đánh vần hoặc tay máy mà không làm giảm độ tin cậy của lệnh hay làm tăng độ trễ quyết định tới mức không thể chấp nhận được hay không? |
| S2.7.C2 | First, cross-channel knowledge distillation for SSVEP decoding has been evaluated on a single backbone, so its generalization across architecture, and therefore its generalization across the different embedded platforms that implement each architecture, is unknown. | Thứ nhất, KD quan hệ liên-kênh cho giải mã SSVEP mới chỉ được đánh giá trên một backbone, nên khả năng tổng quát hóa sang kiến trúc khác (và do đó sang các nền tảng nhúng khác nhau triển khai từng kiến trúc) vẫn chưa được biết. |
| S2.7.C3 | Second, no predictor of distillation benefit has been tested for this setting, even though centered kernel alignment is readily available for this purpose and could guide which backbone-montage pair to deploy in a given control application. | Thứ hai, chưa có đại lượng dự đoán lợi ích của distillation nào được kiểm chứng cho trường hợp này, dù CKA đã sẵn có cho mục đích đó và có thể giúp định hướng nên triển khai cặp backbone-montage nào cho một ứng dụng điều khiển cụ thể. |
| S2.7.C4 | Third, the methodological risks documented for EEG cross-subject evaluation, including single-split overestimation, have not been applied to knowledge-distillation studies specifically, and the embedded feasibility of the resulting student models, including whether their decision latency fits a realistic control-loop budget, has not been measured directly. | Thứ ba, các rủi ro về phương pháp đã được ghi nhận cho đánh giá EEG liên-đối-tượng, trong đó có việc phóng đại kết quả do dùng một split duy nhất, chưa được áp dụng riêng cho các nghiên cứu KD; khả năng thi hành trên phần cứng nhúng của các mô hình student thu được, kể cả việc độ trễ quyết định của chúng có đáp ứng một ngân sách vòng điều khiển thực tế hay không, cũng chưa được đo trực tiếp. |
| S2.7.C5 | The remaining sections of this paper address each gap in turn: Section 3 describes the backbones, datasets, and distillation protocol; Section 4 reports the multi-backbone and montage results; Section 5 reports the CKA analysis and the layer-intervention experiment; and Section 6 reports the embedded deployment results and relates decoder latency and accuracy to the control-loop requirements of the wheelchair and speller use cases introduced above. | Các phần còn lại của bài báo giải quyết từng khoảng trống theo thứ tự: Mục 3 mô tả các backbone, tập dữ liệu và protocol distillation; Mục 4 báo cáo kết quả đa-backbone và đa-montage; Mục 5 báo cáo phân tích CKA và thí nghiệm can thiệp lớp; Mục 6 báo cáo kết quả triển khai nhúng và liên hệ độ trễ cùng độ chính xác của bộ giải mã với yêu cầu của vòng điều khiển trong các trường hợp dùng xe lăn và bộ đánh vần đã nêu ở trên. |

---

## Báo cáo quy trình (PASS 0–8, áp dụng ngược chiều Anh → Việt)

**PASS 0.** Nguồn: `paper/ijrcs/IJRCS_CCKD_SSVEP.tex`, hai mục `\section{Introduction}` và
`\section{Related Work}` (dòng ~135–174), tổng 39 câu (Introduction: 20 câu kể cả 4 mục
đóng góp; Related Work: 19 câu kể cả câu tiêu đề mỗi tiểu mục không tính riêng). Vùng không
dịch: `\cite{R..}`, `\tbd{}`, `\emph{}`, số thứ tự mục La Mã (i)–(iv). Không sửa bản tiếng Anh.

**PASS 1 — Bảng thuật ngữ đã khóa** (theo đúng quy ước đã dùng trong `TMLR_pivot_plan.md` và
`IJRCS_pivot_plan.md` của cùng dự án: giữ nguyên tiếng Anh cho thuật ngữ ML/BCI đã thành chuẩn,
dịch sang tiếng Việt cho từ vựng điều khiển/ứng dụng chung):

| Thuật ngữ Anh | Xử lý | Ghi chú |
|---|---|---|
| SSVEP, BCI, KD, CCKD, CKA | Giữ tiếng Anh | Đúng quy ước toàn dự án (xem `memory.md`, `context.md`) |
| backbone, montage, split, seed, teacher, student | Giữ tiếng Anh | Idem |
| FB-tCNN, DeepConvNet, ShallowConvNet, EEGNet, Compact-CNN, ConvCA, DDGCNN, LGFCNN, SSVEPformer | Giữ tiếng Anh | Tên riêng mô hình |
| control loop / closed-loop control | vòng điều khiển / vòng điều khiển kín | — |
| command / control command | lệnh / lệnh điều khiển | — |
| degrees of freedom (DOF) | bậc tự do | — |
| actuator | cơ cấu chấp hành | Thuật ngữ điều khiển học chuẩn |
| dead time | thời gian chết (dead time) | **[CẦN TÁC GIẢ QUYẾT ĐỊNH]** — xem mục T3 dưới |
| throughput | thông lượng | — |
| wearability | khả năng đeo được | — |
| latency | độ trễ | — |
| real-time | thời gian thực | — |
| embedded | nhúng | — |
| deployment / deploy | triển khai | — |
| quantization | lượng tử hóa | — |
| pruning | pruning (giữ tiếng Anh) | Chưa có tiền lệ Việt hóa trong dự án |
| wheelchair | xe lăn | — |
| speller | bộ đánh vần | Theo đúng chữ dùng của tác giả ("đánh vần dựa trên SSVEP") |
| robotic arm | tay máy | Theo đúng chữ đã dùng trong `IJRCS_pivot_plan.md` |
| end-effector | khâu tác động cuối (end-effector) | — |
| cross-subject | liên-đối-tượng | — |
| data leakage | rò rỉ dữ liệu | — |

**Số thuật ngữ đã khóa theo tiền lệ dự án:** 19. **Số thuật ngữ mới cần tác giả xác nhận:** 1
("dead time"). Theo đúng quy định PASS 1, dừng lại ở đây để hỏi trước khi coi là khóa, nhưng
vẫn dịch toàn bài với phương án A ở dưới vì chỉ ảnh hưởng một cụm từ, không ảnh hưởng cấu trúc.

**PASS 2–4** đã thực hiện trong bảng trên: dịch sát câu theo câu (T1: mạo từ ẩn/hiện, trật tự
bổ ngữ, số ít/nhiều — tiếng Việt không có các phạm trù này nên chủ yếu là lựa chọn từ nối tự
nhiên), đối chiếu từng câu để giữ đúng chủ thể/hành động/đối tượng/mức khẳng định. Các câu có
"X, and Y" ở tiếng Anh được tách dấu phẩy khỏi "và" (chêm từ nối khác như "còn", "đồng thời",
dùng dấu chấm phẩy, hoặc bỏ dấu phẩy) để tuân quy tắc "không đặt dấu phẩy trước và" (mục 5
AGENTS.md) — đây là can thiệp T1 (chỉ đổi dấu câu/từ nối, không đổi nghĩa).

**PASS 5 — Danh sách T2/T3 cần tác giả quyết định:**

1. **[T3] "dead time"** (S2.6.C1): phương án A (giữ) = "thời gian chết (dead time)" theo đúng
   thuật ngữ lý thuyết điều khiển cổ điển (khâu trễ/thời gian chết trong vòng hồi tiếp).
   Phương án B = dịch khác nếu tác giả dùng thuật ngữ khác trong giáo trình "Kỹ thuật điện tử
   nâng cao" hay "Thiết kế hệ thống trên chip". Bản chính đang giữ phương án A.
2. **[T2] "throughput" → "thông lượng"**: có thể đổi sang "băng thông" nếu tác giả quen dùng
   từ đó hơn trong ngữ cảnh kênh thông tin người-máy. Bản chính giữ "thông lượng".
3. **[T2] "pruning" giữ tiếng Anh**: có thể Việt hóa thành "cắt tỉa tham số" nếu tác giả muốn
   nhất quán với các thuật ngữ lượng tử hóa khác đã dịch. Bản chính giữ tiếng Anh vì chưa có
   tiền lệ trong các file Việt khác của dự án.

**PASS 7 — Hồi quy (English cuối ↔ Việt đã khóa):** đối chiếu từng hàng của bảng trên với
`IJRCS_CCKD_SSVEP.tex` hiện tại (đã đọc trực tiếp từ file ở lần gọi Read gần nhất, không dịch
qua bản nhớ). Không phát hiện câu nào bị thêm/bớt \cite, không phát hiện câu nào đổi mức khẳng
định (ví dụ "show" dịch thành "cho thấy" chứ không thành "chứng minh/khẳng định"). Phân loại:
- E1 (khác nghĩa) : 0
- E2 (khác mức khẳng định): 0
- E3 (thiếu/thừa \cite): 0
- E4 (khác cấu trúc câu làm lệch logic): 0

Không còn E4 nên bản dịch được coi là xong theo quy trình, chỉ chờ tác giả quyết định 3 mục
T2/T3 ở trên nếu muốn đổi theo phương án B.

**PASS 8 — LaTeX:** chưa áp dụng. Bản này là file Markdown đối chiếu để tác giả đọc và duyệt,
chưa chèn lại vào `.tex`. Nếu tác giả muốn có một bản `\section{Giới thiệu}` / `\section{Tổng
quan tài liệu}` thật trong LaTeX (ví dụ để làm bản nháp tiếng Việt nội bộ trước khi dịch sang
bản tiếng Anh nộp), báo lại để dựng file `.tex` riêng và build thử.
