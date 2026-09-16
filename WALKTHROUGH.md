# BÁO CÁO TOÀN DIỆN DỰ ÁN NOISY CAUSAL
## Chi Phí Của Cấu Trúc Nhân Quả Sai, Neo Từ Vựng & Bản Chất Nhận Thức Của LLM
> *(Bản tổng hợp chi tiết từ Phương pháp luận đến Số liệu thực nghiệm — Trình bày bằng Markdown thuần, giải thích tường minh mọi khái niệm cho người mới bắt đầu)*

---

## TỪ ĐIỂN KHÁI NIỆM DÀNH CHO NGƯỜI MỚI BẮT ĐẦU

Trước khi đi vào phương pháp và số liệu, dưới đây là các khái niệm nền tảng được giải thích bằng trực giác đời thường:

* **LLM (Large Language Model - Mô hình ngôn ngữ lớn):** Các hệ thống trí tuệ nhân tạo như GPT-4, GPT-4.1, Claude, Llama. Về bản chất, chúng được huấn luyện trên hàng nghìn tỷ từ ngữ từ Internet để làm nhiệm vụ: **đoán từ tiếp theo có xác suất xuất hiện cao nhất** dựa trên chuỗi từ ngữ phía trước.
* **Tương quan (Correlation) vs. Nhân quả (Causation):** 
  * *Tương quan:* Hai sự việc thường xuất hiện cùng lúc. Ví dụ: Khi trời mùa hè nắng gắt, lượng kem bán ra tăng mạnh và số vụ đuối nước cũng tăng mạnh. Hai hiện tượng này tương quan thống kê với nhau.
  * *Nhân quả:* Sự việc này là nguyên nhân trực tiếp sinh ra sự việc kia. Thời tiết nắng nóng là nguyên nhân khiến người ta đi bơi nhiều (dẫn đến đuối nước) và ăn kem nhiều; nhưng ăn kem **không phải** nguyên nhân gây ra đuối nước.
* **Đồ thị nhân quả (Causal DAG - Directed Acyclic Graph):** Một sơ đồ gồm các nút tròn (biến/sự kiện) và các mũi tên nối giữa chúng. Mũi tên `A → B` biểu thị "$A$ trực tiếp gây ra $B$". "Phi chu trình" nghĩa là các mũi tên đi một chiều, không bao giờ quay ngược lại tạo thành vòng luẩn quẩn (`A → B → C → A`).
* **Can thiệp (Intervention) và Toán tử do(X):** 
  * *Quan sát thông thường:* Bạn thấy một người tự nguyện uống thuốc và bệnh thuyên giảm.
  * *Can thiệp [do(X)]:* Bác sĩ chủ động chỉ định bắt buộc bệnh nhân phải uống thuốc (cắt đứt mọi yếu tố ngoại cảnh) để đo lường chính xác hiệu quả chữa bệnh của viên thuốc.
* **Điểm phần trăm (pp - Percentage Points):** Thước đo chênh lệch tuyệt đối giữa hai tỷ lệ phần trăm. 
  * Ví dụ: Điểm số tăng từ 70% lên 80% nghĩa là tăng **10 điểm phần trăm (10 pp)**. Nếu nói "tăng 10%" là sai về toán học (vì 10% của 70% chỉ là 7% → lên 77%).
* **Khoảng tin cậy 95% (95% Confidence Interval - viết tắt là CI [a ; b]):** Khi làm thí nghiệm trên một mẫu câu hỏi, con số thu được luôn có dao động ngẫu nhiên. Khoảng tin cậy `[a ; b]` cho biết: Ta tin tưởng 95% rằng giá trị thực tế của toàn bộ tổng thể nằm trong khoảng từ `a` đến `b`.
  * *Quy tắc suy diễn:* Nếu khoảng tin cậy của một hiệu số chứa số 0 (ví dụ `[-1,5 ; +2,0]`), ta **chưa thể khẳng định có sự khác biệt thực sự**; kết quả đó hoàn toàn có thể do may rủi.
* **Giá trị p (p-value):** Xác suất để một kết quả tốt xảy ra hoàn toàn do ăn may ngẫu nhiên. Quy chuẩn khoa học quy định: nếu **p < 0,05 (dưới 5%)**, kết quả mới được công nhận là "có ý nghĩa thống kê" (đáng tin cậy).
* **Difference-in-Differences (DiD - Hiệu của các hiệu số):** Phương pháp đo lường mức độ tác động thuần túy của một can thiệp.
  * Công thức: `DiD = (Tác hại khi KHÔNG có đồ thị) - (Tác hại khi CÓ đồ thị)`
  * Nếu `DiD > 0`: Cấp đồ thị giúp làm giảm bớt tổn thất do việc ẩn danh hóa gây ra.

---

# PHẦN I: TỔNG QUAN, BỐI CẢNH & CÂU HỎI NGHIÊN CỨU

### 1.1. Con số 50–70% trên benchmark và Nghịch lý quan sát
Trên các bộ đề kiểm tra suy luận nhân quả (như benchmark CLadder), các mô hình AI tiên tiến thường đạt độ chính xác từ **50% đến 70%**. Con số này dẫn đến hai cách giải thích hoàn toàn trái ngược:
1. **Tư duy cấu trúc (Structural Reasoning):** Mô hình thực sự tiếp nhận đồ thị, duyệt theo các chiều mũi tên, tính toán can thiệp logic để ra đáp án. Tên biến chỉ là nhãn đại diện.
2. **Ghi nhớ từ vựng bề mặt (Lexical Memory):** Mô hình hoạt động như một con vẹt ngẫu nhiên. Vì đã đọc hàng triệu văn bản trên mạng, nó nhớ cụm từ *"hút thuốc"* thường đi cùng *"ung thư"*. Nó trả lời đúng chỉ nhờ nhớ từ ngữ quen thuộc.

**Nghịch lý quan sát (Observational Equivalence):** Trên các bài toán thông thường, cấu trúc nhân quả đúng và từ ngữ đời thường luôn đi đôi với nhau (*"Nghèo đói dẫn đến nước bẩn, nước bẩn gây ra dịch tả"*). Dù dùng tư duy thật hay học vẹt, mô hình đều cho ra đáp án đúng. Do đó, điểm số 50–70% **hoàn toàn không cho biết mô hình dùng cơ chế nào**.

---

### 1.2. Thao tác ẩn danh hóa của Caliper (arXiv:2606.04915)
Để phân ly hai cơ chế trên, bài báo Caliper (tháng 6/2026) giữ nguyên 100% cấu trúc logic và các bảng xác suất, nhưng **ẩn danh hóa tên biến** (thay tên thật bằng ký hiệu `A, B, C` hoặc từ vô nghĩa `glimx, muvq`).

* **Kết quả của Caliper:** Điểm số của 14 mô hình lớn nhỏ **đồng loạt sụp đổ từ 7,6 đến 29,6 điểm phần trăm**, nhiều mô hình rơi thẳng về mức 50% (đoán mò ngẫu nhiên).
* **Kết luận:** Năng lực của AI phụ thuộc nặng nề vào "chiếc nạng" từ vựng đời thường; khi mất từ vựng, khả năng nhân quả biến mất.

---

### 1.3. Hai khoảng trống nghiên cứu (Research Gaps)
Mặc dù Caliper đã chỉ ra điểm yếu của AI, y văn vẫn bỏ ngỏ hai câu hỏi lớn:

1. **Lỗ hổng từ NoisyCausal (arXiv:2605.04313):** Bài báo này phát hiện rằng cấp đồ thị đúng giúp AI tăng từ 65,32% lên 85,00%, nhưng chỉ cần **đảo chiều 1 cạnh**, mô hình tụt ngay xuống 73,20% (-11,8 pp); nếu cấp đồ thị ngẫu nhiên, mô hình tụt về 60,87% (tệ hơn cả lúc không có đồ thị).
   * *Khoảng trống 1:* Đồ thị sai gây "độc" cho mô hình. Nhưng NoisyCausal không trả lời: **Đồ thị phải sai bao nhiêu cạnh thì việc dùng nó hết có lãi? Điểm hòa vốn nằm ở đâu?**
2. **Lỗ hổng từ Caliper (arXiv:2606.04915):** Caliper chứng minh AI sụp đổ khi mất từ vựng, nhưng họ chỉ bắt mô hình tự vẽ lại cạnh (vốn bất khả thi khi không hiểu từ vựng). Họ chưa từng cấp đồ thị nhân quả hoàn chỉnh cho mô hình.
   * *Khoảng trống 2:* Nếu mô hình bị tước mất chiếc nạng từ vựng, **việc cấp sẵn một đồ thị nhân quả tường minh có bù đắp lại được phần năng lực đã mất không?** Cấu trúc logic có thực sự thay thế được tri thức thế giới?

---

### 1.4. Hai câu hỏi nghiên cứu cốt lõi của đề tài
* **Câu hỏi 1 (Q1 - Định giá lỗi cấu trúc & Điểm hòa vốn):** Mỗi loại lỗi đồ thị (đảo chiều cạnh `DR`, xóa thiếu cạnh `ED`, thêm thừa cạnh `FE`) làm suy giảm bao nhiêu điểm phần trăm? Điểm hòa vốn `k*` (Break-even point - số cạnh sai tối đa trước khi đồ thị gây hại hơn là không dùng) là bao nhiêu?
* **Câu hỏi 2 (Q2 - Bản chất nhận thức & Tính thay thế):** Năng lực biểu kiến của LLM là năng lực từ vựng hay năng lực cấu trúc? Cấp đồ thị đúng có khôi phục được tổn thất do ẩn danh hóa gây ra hay không?

---

# PHẦN II: PHƯƠNG PHÁP LUẬN & THIẾT KẾ THỰC NGHIỆM

### 2.1. Nền tảng dữ liệu benchmark CLadder & Bộ kiểm chứng giải tích
Dự án không tự sinh dữ liệu bằng LLM (tránh bẫy ảo giác) mà xây dựng hoàn toàn trên benchmark **CLadder** (Jin et al., NeurIPS 2023). Nhãn của CLadder được sinh từ các Mô hình Nhân quả Cấu trúc (SCM) hình thức.

* **Kiểm chứng giải tích độc lập:** Nhóm nghiên cứu đã tự lập trình bộ giải toán giải tích SCM, tính toán lại toàn bộ 66.824 phép tính trên 7.064 mô hình của CLadder.
* **Độ chính xác:** Khớp tuyệt đối với lý thuyết ở sai số máy tính: **6,66 x 10⁻¹⁶**.

---

### 2.2. Thiết kế ghép cặp trong cùng một câu hỏi (Within-item Paired Design)
Mọi so sánh đều thực hiện trên **cùng một bài toán**: Cùng cấu trúc đồ thị, cùng các con số xác suất, cùng câu hỏi và nhãn chuẩn. Mô hình giải cùng một bài toán dưới các điều kiện can thiệp khác nhau để loại bỏ 100% sai số ngẫu nhiên giữa các đề bài.

---

### 2.3. Kỹ thuật phẫu thuật gỡ sạch đồ thị văn xuôi
Prompt gốc của CLadder luôn lồng sẵn đồ thị vào văn bản:  
`"Poverty has a direct effect on water quality and cholera. Water company has a direct effect on water quality..."`

* **Nguy cơ:** Nếu để nguyên, điều kiện "không đồ thị" thực chất vẫn chứa đồ thị lén lút bên trong; còn cấp thêm đồ thị sai sẽ biến thành bài toán "xung đột văn cảnh".
* **Giải pháp:** Sử dụng biểu thức chính quy (Regex) quét mẫu chữ `has a direct effect on` để bóc tách triệt để toàn bộ các câu mô tả cạnh. Thân đề bài sạch chỉ còn bối cảnh, các con số xác suất và câu hỏi. Đồ thị chỉ được đưa vào ở một khối cấu trúc riêng biệt ở cuối prompt nếu điều kiện yêu cầu.

---

### 2.4. Ma trận thực nghiệm 2 trục can thiệp

Dự án thiết lập ma trận 2 trục can thiệp độc lập trên cùng một tập câu hỏi:

```
[TRỤC TỪ VỰNG: Mức độ cung cấp tri thức đời thường]
  KEEP         ──► Tên biến gốc đời thường hợp lẽ tự nhiên (Poverty, Cholera)
  IRRELEVANT   ──► Tên đồ gia dụng có thật ngoài đời (Spoon, Lamp, Kettle)
  PERMUTE      ──► Hoán vị chính các từ đó sang chiều SAI (Cholera causes Poverty)
  SYMBOL       ──► Ký hiệu trừu tượng ngắn gọn (Biến A, Biến B, Biến C)
  PSEUDO       ──► Từ giả vô nghĩa nhiều âm tiết của CLadder (Glimx, Muvq, Zuph)

[TRỤC CẤU TRÚC: Mức độ đồ thị cung cấp cho mô hình]
  RAW          ──► Thân đề bài đã gỡ sạch đồ thị (Đường sàn thực tế không có đồ thị)
  RAW_INSTR    ──► Thân đề bài + Câu lệnh nhắc suy luận nhân quả, KHÔNG có đồ thị
  ORACLE       ──► Thân đề bài + Khối đồ thị chuẩn xác 100%
  DR_k1        ──► Thân đề bài + Khối đồ thị bị ĐẢO CHIỀU 1 cạnh
  DR_k2, DR_k3 ──► Thân đề bài + Khối đồ thị bị đảo chiều 2 hoặc 3 cạnh
  ED_k, FE_k   ──► Thân đề bài + Khối đồ thị bị XÓA BỚT cạnh hoặc THÊM THỪA cạnh
```

---

### 2.5. Phân tầng 3 nhóm câu hỏi trong Benchmark
Mẫu câu hỏi CLadder trộn lẫn 3 nhóm bài toán có bản chất nhận thức rất khác nhau:
1. **Nhóm số học thuần (32,2% mẫu - marginal, correlation):** Đề bài đã cho sẵn toàn bộ xác suất, chỉ cần tính toán số học. Đồ thị không đóng vai trò gì.
2. **Nhóm nhận dạng cấu trúc (18,4% mẫu - backadj):** Đề bài hỏi trực tiếp: *"Tập biến nào sau đây là tập hiệu chỉnh hợp lệ?"*. Với câu hỏi này, **đồ thị chính là đáp án**. Cấp đồ thị cho mô hình giống như cho xem trước đáp án, làm phồng điểm số một cách giả tạo.
3. **Nhóm suy luận nhân quả thực sự (49,4% mẫu - nòng cốt là ate):** Đồ thị là công cụ trung gian bắt buộc để tìm đường điều chỉnh trước khi tính toán can thiệp. **Đây là nhóm duy nhất đo lường đúng mục tiêu nghiên cứu.**

---

# PHẦN III: KẾT QUẢ THỰC NGHIỆM CHẠY THỰC TẾ

Dưới đây là toàn bộ kết quả số liệu thu được từ hơn **60.000 lượt gọi API** trên 3 mô hình thuộc họ GPT-4.1 (`gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`):

---

### 3.1. Bảng độ chính xác cơ bản trên các điều kiện (n = 174 item)

Bảng dưới đây thể hiện độ chính xác (%) trên các câu đọc được câu trả lời:

| Mô hình | Điều kiện cấu trúc | KEEP (Từ thật) | PERMUTE (Từ hoán vị) | SYMBOL (Ký hiệu A,B) | PSEUDO (Từ giả) |
|---|---|:---:|:---:|:---:|:---:|
| **gpt-4.1** | RAW (Không đồ thị) | 75,58% | 65,50% | 67,24% | 66,28% |
| **gpt-4.1** | ORACLE (Đồ thị đúng) | **87,13%** | **78,61%** | **78,95%** | **79,77%** |
| **gpt-4.1-mini** | RAW (Không đồ thị) | 77,91% | 64,91% | 70,52% | 64,37% |
| **gpt-4.1-mini** | ORACLE (Đồ thị đúng) | **84,80%** | **83,64%** | **82,35%** | **81,98%** |
| **gpt-4.1-nano** | RAW (Không đồ thị) | 79,19% | 71,08% | 64,16% | 67,24% |
| **gpt-4.1-nano** | ORACLE (Đồ thị đúng) | 74,25% | 61,88% | 68,45% | 68,86% |

**Quan sát quan trọng:**
* Với hai mô hình mạnh (`gpt-4.1` và `mini`), cấp đồ thị đúng `ORACLE` giúp phục hồi điểm số trên tất cả các nhánh ẩn danh từ vựng từ mức ~65% lên xấp xỉ 80–83%.
* Với mô hình nhỏ nhất (`gpt-4.1-nano`), cấp đồ thị đúng ở nhánh `KEEP` lại làm điểm số **tụt từ 79,19% xuống 74,25%** (mất -4,94 pp). Mô hình yếu nhất làm bài tệ hơn khi bị ép dùng đồ thị.

---

### 3.2. Hiệu ứng Difference-in-Differences (DiD) trên nhóm câu hỏi nhân quả

Khi lọc riêng nhóm câu hỏi nhân quả thực sự (loại bỏ nhóm số học thuần và nhóm nhận dạng `backadj`), đại lượng `DiD` đo lường mức độ thu hẹp khoảng cách tổn thất do ẩn danh:

```
DiD = (Tác hại khi KHÔNG có đồ thị) - (Tác hại khi CÓ đồ thị)
```

Kết quả đo lường qua 3 tập mẫu độc lập và mẫu gộp:

| Tập mẫu | Số câu hỏi nhân quả | Mức DiD đo được | Khoảng tin cậy 95% (CI) | Giá trị p | Đánh giá |
|---|:---:|:---:|:---:|:---:|---|
| **Mẫu khám phá ban đầu** | n = 85 câu | **+12,71 pp** | [+2,11 ; +23,11] | p = 0,018 | Đạt ý nghĩa |
| **Mẫu mở rộng đợt 2** | n = 287 câu | **+6,78 pp** | [+1,70 ; +11,85] | p = 0,009 | Đạt ý nghĩa |
| **Mẫu mở rộng đợt 3** | n = 195 câu | **+1,72 pp** | [-5,48 ; +8,92] | p = 0,642 | Trượt ý nghĩa |
| **TỔNG HỢP GỘP (bỏ trùng id)** | **n = 490 câu** | **+5,98 pp** | **[+1,78 ; +10,30]** | **p = 0,005** | **Xác lập vững chắc** |

**Phân tích sâu:**
* Hiệu ứng thu hẹp tổn thất thực sự tồn tại: Cấp khối cấu trúc giúp lấy lại trung bình **+5,98 điểm phần trăm** (p = 0,005).
* Con số tiêu đề ban đầu (+14,35 pp ở mẫu nhỏ $n=86$) thực chất là ước lượng của mẫu khám phá nhỏ. Khi mở rộng ra 490 câu hỏi, hiệu ứng thực tế ổn định ở mức khoảng **+6 điểm phần trăm** (hiện tượng co hẹp hiệu ứng - Winner's Curse).

---

### 3.3. Thực nghiệm với Đồ thị SAI: Đồ thị có cần ĐÚNG không?

Để kiểm tra xem hiệu ứng trên có thực sự đến từ việc mô hình hiểu đúng cấu trúc logic hay không, nhóm thực hiện can thiệp thay thế đồ thị đúng (`ORACLE`) bằng một đồ thị **bị đảo ngược 1 cạnh (`DR_k1`)**:

| Loại cấu trúc cấp cho mô hình | Mức DiD đạt được | Khoảng tin cậy 95% (CI) | Giá trị p |
|---|:---:|:---:|:---:|
| **Đồ thị ĐÚNG (ORACLE)** | **+14,31 pp** | [+6,30 ; +22,52] | p = 0,001 |
| **Đồ thị đúng nằm trong văn xuôi (PROSE)** | **+13,91 pp** | [+7,30 ; +20,47] | p < 0,001 |
| **Đồ thị SAI 1 cạnh (DR_k1)** | **+8,64 pp** | **[+1,81 ; +16,38]** | **p = 0,018** |

* **So sánh trực tiếp giữa Đồ thị đúng và Đồ thị sai:**
  * Hiệu số `ORACLE - DR_k1` = **+1,62 pp** (CI 95%: `[-2,07 ; +5,30]`, p = 0,40).
* **KẾT LUẬN CỐT LÕI:**
  * Khoảng tin cậy của hiệu số chứa số 0 và p = 0,40: **Không có sự khác biệt có ý nghĩa thống kê giữa đồ thị đúng và đồ thị sai 1 cạnh**.
  * Đồ thị sai 1 cạnh vẫn đạt mức tăng +8,64 pp (p = 0,018), hoàn thành được **60% công việc** của đồ thị đúng.
  * Vì vậy, nghiên cứu khẳng định: **Chưa có bằng chứng cho thấy khối cấu trúc cấp cho mô hình bắt buộc phải ĐÚNG**.

---

### 3.4. Phân rã hai chiều: Nâng nhánh yếu hay kéo tụt nhánh mạnh?

Khi mổ xẻ xem tại sao khoảng cách tổn thất giữa hai nhánh từ vựng lại được thu hẹp khi cấp đồ thị:

```
Tác động của việc cấp đồ thị:
  • Nhánh ẩn danh tên biến:       Được NÂNG LÊN   +9,75 pp
  • Nhánh từ vựng đời thường:     Bị KÉO TỤT XUỐNG -4,60 pp
  ---------------------------------------------------------
  Hiệu số tương tác DiD gộp:                      +14,35 pp
```

Bảng phân rã chi tiết theo từng mô hình trên nhánh từ vựng đời thường (`KEEP`):

| Mô hình | Hiệu năng khi KHÔNG đồ thị (RAW) | Hiệu năng khi CÓ đồ thị (ORACLE) | Mức độ thay đổi |
|---|:---:|:---:|:---:|
| **gpt-4.1-nano** | 79,19% | 66,80% | **-12,39 pp (Kéo tụt rất nặng)** |
| **gpt-4.1-mini** | 77,91% | 70,23% | **-7,68 pp (Kéo tụt)** |
| **gpt-4.1** | 75,58% | 81,85% | **+6,27 pp (Nâng lên)** |

* **Ý nghĩa:** **32% hiệu ứng thu hẹp khoảng cách là do khối cấu trúc làm hại điều kiện vốn đang chạy tốt!** Ở hai mô hình cỡ nhỏ và vừa (`nano` và `mini`), việc đưa thêm khối cấu trúc gây ra hiện tượng quá tải thông tin, khiến mô hình bị lú lẫn và phán đoán tệ hơn là để nó tự làm bài bằng trực giác từ vựng.

---

### 3.5. Kiểm tra vai trò của câu lệnh nhắc đơn thuần (RAW_INSTR)

Để kiểm tra xem hiệu ứng có phải chỉ đơn giản là do câu lệnh nhắc nhở *"Hãy dùng cấu trúc nhân quả khi suy luận"* hay không, nhóm chạy điều kiện `RAW_INSTR` (có câu lệnh nhắc nhưng **không có bất kỳ đồ thị nào**):

| Phép so sánh can thiệp | Mức DiD đạt được | Khoảng tin cậy 95% (CI) | Giá trị p |
|---|:---:|:---:|:---:|
| A. RAW so với ORACLE (Cấp đồ thị đầy đủ) | +12,71 pp | [+2,11 ; +23,11] | p = 0,019 |
| B. Chỉ có câu lệnh nhắc, KHÔNG đồ thị | +0,10 pp | [-6,40 ; +6,34] | p = 1,000 |
| C. Có đồ thị khi câu lệnh đã nằm sẵn ở nền | **+13,48 pp** | **[+4,27 ; +22,95]** | **p = 0,003** |

* **Kết luận:** Bản thân câu lệnh nhắc đơn thuần không tạo ra hiệu ứng (chỉ đạt +0,10 pp, CI chứa 0). Phải có sự xuất hiện của khối danh sách cạnh đồ thị thì hiệu ứng mới đạt +13,48 pp (p = 0,003).

---

### 3.6. Bậc thang từ vựng 5 bậc: Đo lường chính xác thứ bị mất đi

So sánh sự sụt giảm độ chính xác khi bước qua từng bậc từ vựng:

| Bậc chuyển đổi | Yếu tố duy nhất bị thay đổi | Mức chênh lệch độ chính xác | Đánh giá thống kê |
|---|---|:---:|---|
| **KEEP → IRRELEVANT** | **Mất tri thức đời thường hợp lẽ** (chuyển sang đồ gia dụng) | **-17,67 pp** | **CI [-26,53 ; -9,24], p < 0,0001 (Xác lập rất vững)** |
| **IRRELEVANT → PERMUTE** | Bị gán thêm một tri thức **SAI** | Dưới 8,5 pp | Chưa tách khỏi 0 (không có ý nghĩa) |
| **IRRELEVANT → SYMBOL** | Từ thật chuyển thành ký hiệu A, B | Dưới 6,4 pp | Chưa tách khỏi 0 (không có ý nghĩa) |
| **SYMBOL → PSEUDO** | Ký hiệu chuyển thành từ giả vô nghĩa | Dưới 6,3 pp | Chưa tách khỏi 0 (không có ý nghĩa) |

* **Kết luận nhận thức:** **Khoảng 90% tổn thất khi ẩn danh hóa xuất phát từ việc đánh mất tri thức đời thường hợp lẽ tự nhiên**. Khi tri thức đời thường đã mất, việc tên biến là từ gia dụng có nghĩa, ký hiệu một chữ cái hay từ giả vô nghĩa đều gây hại ngang nhau.

---

### 3.7. Vị trí hiệu ứng sinh sống: Kích thước đồ thị & Loại bài toán

Hiệu ứng thu hẹp tổn thất không phân bố rải rác mà tập trung cục bộ ở các bài toán phức tạp:

| Phân nhóm dữ liệu | Số câu hỏi (n) | Mức DiD đạt được | Giá trị p | Đánh giá |
|---|:---:|:---:|:---:|---|
| **Đồ thị nhỏ (3 nút)** | n = 40 câu | +5,72 pp | p = 0,217 | Không có ý nghĩa |
| **Đồ thị lớn (≥ 4 nút)** | n = 46 câu | **+22,44 pp** | **p = 0,0005** | **Hiệu ứng cực mạnh** |
| **Bài toán can thiệp ATE** | n = 24 câu | **+27,44 pp** | **p = 0,0005** | **Hiệu ứng cực mạnh** |
| **Bài toán phản thực tế ETT** | n = 21 câu | +2,48 pp | p = 0,760 | Không có ý nghĩa |

* **Ý nghĩa:** Càng có nhiều biến xuất hiện trong đồ thị (≥ 4 nút) và bài toán càng đòi hỏi nhiều bước tính toán trung gian (`ate`), khối cấu trúc càng phát huy tác dụng rõ rệt.

---

### 3.8. Phân tích chuỗi suy luận (Chain-of-Thought): Đồ thị thâm nhập vào đâu?

Khi kiểm tra điều kiện đồ thị bị đảo chiều 1 cạnh (`DR_k1`), nhóm đọc chi tiết từng dòng lập luận mà mô hình viết ra để xem mô hình có thực sự dùng chiều mũi tên bị sai hay không:

* **Tần suất lặp lại chiều mũi tên được cấp:**
  * `gpt-4.1-mini`: **57,5%** số lần mô hình viết ra chiều mũi tên sai theo đúng đồ thị được cấp.
  * `gpt-4.1`: **43,1%** số lần.
  * `gpt-4.1-nano`: **34,5%** số lần.
  * *So sánh với đường sàn:* Khi hoàn toàn không cấp đồ thị, mô hình tự nói về chiều cạnh chỉ từ **2,3% đến 4,6%**. Điều này chứng minh khối cấu trúc là thứ ép mô hình phải phát biểu về chiều mũi tên.
* **Nghịch lý tính toán:**
  * Trong số các câu mà mô hình phát biểu bằng lời **SAI CHIỀU MŨI TÊN** trong lời giải, có tới **75% đến 82% trường hợp kết quả tính toán cuối cùng vẫn ra ĐÚNG** y hệt như khi được cấp đồ thị chuẩn!
  * **Giải thích:** Chiều mũi tên chỉ đi vào phần văn bản phát biểu; nó chỉ thực sự làm thay đổi phép tính số học bên trong ở khoảng **1/4 trường hợp**. Đây là lý do giải thích tại sao đồ thị sai 1 cạnh vẫn giúp mô hình đạt điểm số gần ngang đồ thị đúng.

---

### 3.9. Kết quả trích xuất đồ thị (Graph Induction)

Khi yêu cầu mô hình đọc văn bản và tự dựng lại đồ thị:
* **Đường sàn đoán mò ngẫu nhiên:** Trên đồ thị nhỏ 3–5 nút của CLadder, một thuật toán bốc cạnh ngẫu nhiên đạt điểm F1 = **0,362**.
* **Hiệu năng tự dựng của mô hình:**
  * Nhánh từ thật (`KEEP`): Đạt F1 = **0,462 đến 0,608** (vượt rõ rệt trên sàn ngẫu nhiên).
  * Nhánh hoán vị (`PERMUTE`): Đạt F1 = **0,384 đến 0,424** (chỉ nhỉnh hơn mức đoán ngẫu nhiên từ 0,02 đến 0,06).
* **Số cạnh bị đảo ngược trên mỗi bài toán:**
  * Khi có prior đúng (`KEEP`): Chỉ có **0,046 đến 0,069** cạnh bị đảo ngược.
  * Khi prior vắng mặt (`SYMBOL`, `PSEUDO`): Có **0,092 đến 0,155** cạnh bị đảo ngược (tăng 1,3 đến 2,8 lần).
  * Khi bị gán prior SAI (`PERMUTE`): Có tới **0,316 đến 0,529** cạnh bị đảo ngược (**tăng vọt 5,1 đến 7,7 lần**).
* **Kết luận trích xuất:** Khi tên biến gợi ý một chiều nhân quả sai, mô hình bị thiên kiến tri thức đời thường dẫn dắt: Nó đi theo tri thức cũ khoảng **33% đến 55% quãng đường** thay vì đọc đúng chiều quan hệ đã nêu trong đề bài.

---

### 3.10. Định giá từng loại lỗi đồ thị & Điểm hòa vốn

Thử nghiệm gây lỗi k cạnh (k = 1, 2, 3) trên hai nhánh từ vựng (`KEEP` và `PSEUDO`) với n = 400 item:

* **Vế Giá của một cạnh sai (Price per error):**
  * Giá của một cạnh đảo chiều `DR`: Làm mất khoảng **4,09 pp/cạnh** ở `gpt-4.1`.
  * Chênh lệch giá giữa tên thật (`KEEP`) và từ giả (`PSEUDO`):
    * `gpt-4.1`: **-0,11 pp** (CI 95%: `[-1,62 ; +1,42]`).
    * `gpt-4.1-mini`: **+0,18 pp** (CI 95%: `[-1,53 ; +2,09]`).
  * *Kết luận vế giá:* Giá của một cạnh sai **không thay đổi theo miền từ vựng**, với cận tương đương dưới 1,6 pp/cạnh.
* **Vế Ngân sách và Điểm hòa vốn k*:**
  * Trên tập mẫu gộp $n \approx 800$, ngân sách (`ORACLE - RAW`) đạt **+4,67 pp** ở `gpt-4.1` và **+4,38 pp** ở `mini`.
  * Tuy nhiên, trên nhóm câu hỏi nhân quả thực sự, ngân sách bị âm ở 2/3 mô hình. Do đó, công thức điểm hòa vốn `k* = ngân sách / giá` **chưa xác lập được về mặt khoa học**.

---

### 3.11. Phát hiện chấn động về Benchmark CLadder (NeurIPS 2023)

Trong quá trình đối chiếu giải tích 66.824 phép tính trên 7.064 mô hình SCM của CLadder, dự án phát hiện một sai sót cấu trúc trong bài báo NeurIPS 2023 gốc:

* **Lỗi toán học:** Khi tính toán xác suất can thiệp, thuật toán của CLadder đã **nhân xác suất biên của các nút cha lại với nhau như thể chúng độc lập** ở 7/10 họ đồ thị:
  ```
  Công thức tính của CLadder: P(Pa_Y) = Tích các xác suất P(V) với mọi V thuộc Pa_Y
  ```
  Trong khi trên thực tế, các nút cha này có liên kết phụ thuộc lẫn nhau qua các đường dẫn khác.
* **Hậu quả trên nhãn dữ liệu:** Lọc riêng các câu hỏi mà sai số toán học này làm giá trị xác suất dịch chuyển qua ngưỡng phân loại 0,5 (làm đổi đáp án từ Có sang Không):
  * **82 trên 85 câu hỏi có đáp án công bố đi theo giá trị BỊ TÍNH TOÁN SAI của CLadder.**
  * Chỉ có **3 câu hỏi** đi theo giá trị toán học chuẩn xác.
* **Mức độ ảnh hưởng:** Tác động tới khoảng **0,8% số câu** trong mẫu của dự án (ước tính 1,4 câu trên 174 câu). Nhờ thiết kế thực nghiệm ghép cặp lấy hiệu số chênh lệch, sai số này triệt tiêu khỏi các phép đo DiD, nhưng làm sai lệch nhẹ mức độ chính xác tuyệt đối.

---

# PHẦN IV: CƠ CHẾ BẢN CHẤT & BỐN GIẢ THUYẾT ĐÃ LOẠI TRỪ

### 4.1. Cơ chế thực sự: "Tái gắn ký hiệu" (Symbol Re-grounding)
Tổng hợp các bằng chứng thực nghiệm:
1. Đồ thị sai 1 cạnh cũng mang lại hiệu quả gần ngang đồ thị đúng (+8,64 pp so với +14,31 pp).
2. Hiệu ứng sống chủ yếu ở các đồ thị lớn (≥ 4 nút đạt +22,44 pp) và bài toán ATE (+27,44 pp).
3. 75% đến 82% số câu mô hình phát biểu sai chiều cạnh trong lời giải nhưng đáp số cuối cùng vẫn tính đúng.

> **KẾT LUẬN CƠ CHẾ:** Khối cấu trúc không hề dạy mô hình tư duy logic nhân quả hình thức. Thay vào đó, nó đóng vai trò như một **mỏ neo bộ nhớ làm việc (Working Memory Anchor)**: Khối văn bản ở cuối đề bài liệt kê lại toàn bộ tên các biến, giúp mô hình "tái gắn ký hiệu" để không bị mất dấu hay quên biến trong các bài toán tính toán số học nhiều bước.

---

### 4.2. Bốn giả thuyết cạnh tranh đã bị loại trừ bằng thực nghiệm

| Giả thuyết cạnh tranh | Phép kiểm tra thực nghiệm | Kết quả thực tế |
|---|---|---|
| **1. Do độ dài prompt** (Khối đồ thị làm prompt dài hơn nên mô hình chú ý hơn) | So sánh độ dài prompt với độ chính xác trên toàn bộ các điều kiện | **Bị loại trừ.** Prompt dài hơn làm kết quả **tệ hơn** (8 ô âm có ý nghĩa, 0 ô dương). Khối đồ thị thêm 221 ký tự cho `KEEP` nhưng thêm ít hơn cho nhánh ẩn danh; lý thuyết độ dài dự đoán chiều ngược lại với thực tế. |
| **2. Do từ thật còn sót lại** (Khoảng một nửa câu ẩn danh vẫn lọt vài danh từ thật) | Tách riêng nhóm câu sạch hoàn toàn và nhóm câu còn sót danh từ thật | **Bị loại trừ.** Nhóm câu sạch hoàn toàn cho hiệu ứng DiD = **+16,15 pp**, trong khi nhóm còn sót từ thật chỉ đạt **+13,71 pp**. Từ thật còn sót làm **co hẹp** hiệu ứng chứ không tạo ra hiệu ứng. |
| **3. Do một lát cắt may mắn** (Cách chia nhóm truy vấn tình cờ gặp may) | Bốc mẫu ngẫu nhiên 50% dữ liệu, lặp lại 2.000 lần mô phỏng bootstrap | **Bị loại trừ.** Trung bình của 2.000 lần bốc ngẫu nhiên chỉ đạt +4,43 pp; và **0/2.000 lần** chạm tới mức +14,35 pp. Cách phân nhóm theo bản chất truy vấn không phải do ăn may. |
| **4. Do cách chấm điểm** (Chấm các câu không parse được thành câu sai) | Đổi quy tắc chấm điểm: coi câu không parse được là câu trả lời sai | **Bị loại trừ.** Mức DiD từ +14,35 pp chuyển thành +14,60 pp (gần như giữ nguyên, không thay đổi kết luận). |

---

# PHẦN V: TỔNG KẾT, HẠN CHẾ & BƯỚC ĐI TIẾP THEO

### 5.1. Tóm tắt toàn bộ nghiên cứu trong 3 luận điểm
1. **Cấp khối cấu trúc giúp thu hẹp tổn thất do ẩn danh hóa tên biến khoảng +5,98 điểm phần trăm** (CI 95%: [+1,78 ; +10,30], p = 0,005 trên 490 câu hỏi ghép cặp).
2. **Chưa có bằng chứng cho thấy khối cấu trúc đó bắt buộc phải ĐÚNG** (cấp đồ thị vẽ sai 1 cạnh vẫn đem lại +8,64 pp, không khác biệt có ý nghĩa với đồ thị đúng). Đồng thời, 32% hiệu ứng thu hẹp tổn thất thực chất là do khối cấu trúc kéo tụt hiệu năng của điều kiện tên thật đời thường (-4,60 pp).
3. **Bản chất của hiện tượng là Tái gắn ký hiệu (Symbol Re-grounding)** hỗ trợ bộ nhớ làm việc của mô hình trên các đồ thị nhiều biến, chứ không phải mô hình đã sở hữu năng lực tư duy nhân quả trừu tượng.

---

### 5.2. Thí nghiệm quyết định còn lại: Điều kiện `NAMES_ONLY`
Để giải quyết dứt điểm câu hỏi *"Khối cấu trúc giúp mô hình nhờ các mũi tên nhân quả hay chỉ nhờ danh sách liệt kê tên biến?"*:
* **Thiết kế can thiệp:** Giữ nguyên vị trí khối văn bản ở cuối bài, giữ nguyên câu lệnh nhắc, nhưng **chỉ in danh sách tên biến mà không có bất kỳ mũi tên nào**:
  ```
  The variables of this world are: A, B, C, D.
  Use these variables when reasoning.
  ```
* **Kịch bản phán quyết:**
  * *Nếu đồ thị có mũi tên (ORACLE) vượt trội hơn hẳn danh sách chỉ có tên biến (NAMES_ONLY):* Khẳng định AI thực sự biết sử dụng cấu trúc quan hệ nhân quả.
  * *Nếu danh sách chỉ có tên biến (NAMES_ONLY) đạt kết quả ngang ngửa đồ thị có mũi tên:* Khẳng định 100% hiện tượng này chỉ là hỗ trợ trí nhớ làm việc, hoàn toàn không có tư duy nhân quả.

---

### 5.3. Rào cản công bố duy nhất: Đa dạng hóa dòng mô hình
Toàn bộ số liệu hiện tại được chạy trên 3 kích cỡ của dòng **GPT-4.1** (OpenAI). Để hoàn tất một bài báo khoa học chuẩn mực nộp cho các hội nghị hàng đầu (ACL, EMNLP, NeurIPS), nghiên cứu cần bổ sung thử nghiệm trên **ít nhất một họ mô hình mã nguồn mở độc lập** (như Llama-3.1-8B/70B hoặc Qwen-2.5 qua API).
