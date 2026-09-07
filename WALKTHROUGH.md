# WALKTHROUGH - CẨM NANG TOÀN DIỆN VỀ DỰ ÁN NOISY CAUSAL

> **Đề tài:** Chi phí của các loại lỗi đồ thị nhân quả trong suy luận LLM, và mức độ phụ thuộc của năng lực đó vào từ vựng đời thường.  
> **Repository:** Noisy Causal  
> **Cập nhật:** 2026-09-08 (19.901 lượt gọi mô hình, 19,98 USD. Bản này **rút lại** phát hiện tiêu đề của bản trước sau khi phát hiện một lỗi nạp dữ liệu - xem mục 5.1)  

Tài liệu này được biên soạn đầy đủ và trực quan bằng định dạng thuần (Markdown chuẩn, hiển thị tốt 100% trên mọi trình xem không cần plugin LaTeX) để bất kỳ ai khi đọc cũng hiểu rõ: **Đề tài làm về gì? Đã triển khai như thế nào? Kết quả cụ thể ra sao? Những bài học kỹ thuật nào đã rút ra? Và hướng phát triển tiếp theo là gì?**

---

## MỤC LỤC
1. [PHẦN 1: ĐỀ TÀI LÀM VỀ GÌ? (Bối cảnh, Động lực & 2 Câu hỏi lớn)](#phần-1-đề-tài-làm-về-gì-bối-cảnh-động-lực--2-câu-hỏi-lớn)
2. [PHẦN 2: NỀN TẢNG PHƯƠNG PHÁP (Vì sao chọn CLadder thay vì NoisyCausal?)](#phần-2-nền-tảng-phương-pháp-vì-sao-chọn-cladder-thay-vì-noisycausal)
3. [PHẦN 3: KIẾN TRÚC MÃ NGUỒN & DỮ LIỆU](#phần-3-kiến-trúc-mã-nguồn--dữ-liệu)
4. [PHẦN 4: THIẾT KẾ THỰC NGHIỆM CHI TIẾT (7 Điều kiện Paired)](#phần-4-thiết-kế-thực-nghiệm-chi-tiết-7-điều-kiện-paired)
5. [PHẦN 5: KẾT QUẢ THỰC NGHIỆM & CÁC PHÁT HIỆN CỐT LÕI](#phần-5-kết-quả-thực-nghiệm--các-phát-hiện-cốt-lõi)
   - [**5.1 ĐÍNH CHÍNH: lỗi nạp dữ liệu đã rút bỏ phát hiện tiêu đề cũ**](#51-đính-chính-trước-khi-đọc-tiếp-một-lỗi-nạp-dữ-liệu-đã-xoá-bỏ-phát-hiện-chấn-động)
6. [PHẦN 6: SỔ TAY XỬ LÝ 8 CÁI BẪY KỸ THUẬT KINH ĐIỂN](#phần-6-sổ-tay-xử-lý-8-cái-bẫy-kỹ-thuật-kinh-điển)
7. [PHẦN 7: HƯỚNG DẪN TÁI LẬP THỰC NGHIỆM TỪ A ĐẾN Z](#phần-7-hướng-dẫn-tái-lập-thực-nghiệm-từ-a-đến-z)
8. [PHẦN 8: BÀI HỌC VỀ CƠ CHẾ SELF-GATING & LỘ TRÌNH TIẾP THEO](#phần-8-bài-học-về-cơ-chế-self-gating--lộ-trình-tiếp-theo)

---

## PHẦN 1: ĐỀ TÀI LÀM VỀ GÌ? (Bối cảnh, Động lực & 2 Câu hỏi lớn)

### 1.1. Bối cảnh: Khi LLM suy luận nhân quả
Trong các ứng dụng thực tế (y tế, kinh tế, phân tích nguyên nhân - kết quả), Large Language Models (LLM) thường bộc lộ điểm yếu chết người: **nhầm lẫn giữa tương quan (correlation) và nhân quả (causation)**. Mô hình có xu hướng học vẹt các mẫu thống kê bề mặt (ví dụ: thấy người ho hay uống siro ho thì tưởng uống siro gây ra ho).

Để khắc phục, hướng đi phổ biến hiện nay của các AI Agent là **Graph-Guided Reasoning (Suy luận dựa vào đồ thị nhân quả)**: Yêu cầu LLM trích xuất các thực thể và tự dựng đồ thị có hướng phi chu trình (Causal DAG: X -> Y), sau đó suy luận từng bước dựa trên đồ thị này.

### 1.2. Nghịch lý khởi nguồn từ bài báo NoisyCausal (arXiv:2605.04313v1)
Bài báo NoisyCausal (tháng 5/2026) thử nghiệm đưa đồ thị vào prompt và thu được kết quả:

| Điều kiện trong NoisyCausal | Độ chính xác | Ghi chú |
|---|:---:|---|
| **Full Graph-Guided** (Có đồ thị hướng dẫn) | **80.68%** | Tăng vọt so với suy luận thông thường |
| **No Graph** (Không có đồ thị) | **65.32%** | Mức suy luận tự nhiên của mô hình |
| **Random Graph** (Đồ thị bị nối lại ngẫu nhiên) | **60.87%** | **Thấp hơn không có đồ thị 4.45 điểm %!** |
| **Oracle Graph** (Đồ thị chuẩn 100%) | **85.00%** | Trần hiệu năng đo trên 1.000 mẫu |
| **Oracle + 1 cạnh đảo chiều** | **73.20%** | Sụt giảm **-11.8 điểm %** chỉ vì 1 cạnh sai! |

> **VẤN ĐỀ CỐT LÕI:**  
> Đồ thị sai không trung tính — nó **gây độc (toxic/harmful)** trực tiếp cho LLM.  
> Nhưng bài báo NoisyCausal chỉ báo cáo con số đó rồi bỏ qua. Họ không trả lời câu hỏi: **Ranh giới chịu đựng nằm ở đâu?**  
> Trong thực tế, AI Agent phải **tự dựng đồ thị** từ văn bản. Mà đã tự dựng thì chắc chắn sẽ có sai sót (điểm F1 cạnh chỉ đạt 70–85%). Nếu ranh giới chịu đựng quá nhỏ, việc ép Agent dựng đồ thị có thể đang vô tình làm hỏng kết quả của chính nó!

### 1.3. Hai câu hỏi nghiên cứu lớn của đề tài
* **Câu hỏi 1 (Định giá lỗi đồ thị & Điểm hòa vốn):**  
  *Mỗi loại lỗi đồ thị (đảo chiều, thiếu cạnh, thừa cạnh) làm suy giảm bao nhiêu % độ chính xác của LLM? Đồ thị phải sai bao nhiêu cạnh thì việc dùng nó hết có lãi (Break-Even Point k*)?*
* **Câu hỏi 2 (Bản chất nhận thức của LLM):**  
  *LLM thực sự hiểu và suy luận được dựa trên cấu trúc đồ thị trừu tượng, hay nó chỉ đang "học vẹt" từ vựng đời thường quen thuộc?*

---

## PHẦN 2: NỀN TẢNG PHƯƠNG PHÁP (Vì sao chọn CLadder thay vì NoisyCausal?)

Mặc dù ý tưởng xuất phát từ NoisyCausal, dự án quyết định xây dựng thực nghiệm trên **CLadder** (Jin et al., NeurIPS 2023) vì 4 lý do phương pháp luận đanh thép:

```
[HẠN CHẾ CỦA NOISYCAUSAL]                      [ĐIỂM TỰA VỮNG CHẮC CỦA CLADDER]
- Không công khai code hay dữ liệu      ==>    - Mã nguồn & dữ liệu mở 100% (NeurIPS 2023)
- 5 bước sinh dữ liệu đều là prompt     ==>    - Nhãn sinh từ SCM hình thức & Do-calculus
- Lỗi logic: Tiêm Confounder nhưng             - Nhóm đã tự kiểm chứng đạt sai số 6.66e-16
  vẫn chấm theo mô hình sạch                   - Có sẵn variable_mapping de doi tu vung
                                                 ngay trong cung mot item
```

1. **Khả năng tái lập:** NoisyCausal không phát hành bất kỳ thứ gì. CLadder công khai toàn bộ 10.112 câu hỏi.
2. **Kiểm chứng Ground Truth tới mức sai số máy tính:**  
   Trong file `scripts/verify_groundtruth.py`, chúng ta tự viết bộ solver toán học để tính ATE giải tích từ các bảng xác suất CPD của các mô hình CLadder. Đối chiếu với nhãn công bố: **2.184 model, sai lệch tuyệt đối tối đa chỉ là 6.66 × 10^-16** (đúng bằng sai số dấu phẩy động của máy tính).
3. **Giải quyết nghịch lý Confounder Injection (CI):**  
   NoisyCausal tiêm biến ẩn gây nhiễu làm xuất hiện đường backdoor nhưng lại chấm theo clean SCM (phạt mô hình điều chỉnh đúng, thưởng mô hình phớt lờ). Module `src/noise.py` của chúng ta tách rõ:
   * `answer_preserving = True`: Nhiễu không làm đổi đại lượng truy vấn -> giữ nguyên nhãn.
   * `answer_preserving = False` (như `CI_ACTIVE`): Thay đổi bản chất phân phối -> tách riêng, không chấm theo nhãn cũ.
4. **Không tự sinh dữ liệu bừa bãi:**  
   Mọi câu hỏi, xác suất, bối cảnh, nhãn chuẩn đều lấy nguyên bản từ CLadder. Dự án chỉ áp dụng các **phép biến đổi cơ học tất định** (lật chiều cạnh, xóa cạnh, đổi tên biến). **Không có bất kỳ LLM nào tham gia viết nhãn dữ liệu.**

---

## PHẦN 3: KIẾN TRÚC MÃ NGUỒN & DỮ LIỆU

```
Noisy Causal/
├── 2605.04313v1.pdf           # Toàn văn bài báo gốc NoisyCausal
├── REPORT.md                  # Báo cáo kết quả nghiên cứu & số liệu chi tiết
├── REVIEW.md                  # Phan bien hoi dong 5 ghe (academic-paper-reviewer)
├── WALKTHROUGH.md             # File cẩm nang hướng dẫn này
├── pyproject.toml             # Cấu hình dependency & linter
├── .env                       # Chứa OPENAI_API_KEY (gitignored)
│
├── data/                      # Dữ liệu benchmark CLadder v1 và v1.5
│   ├── cladder-questions.json        # 10.560 câu hỏi v1 kèm metadata
│   ├── cladder-meta.json             # 7.064 mô hình nhân quả (DAG + CPD + ATE)
│   ├── full_v1.5_default.csv         # 10.112 cau hoi DAY DU - file DUY NHAT dung duoc
│   ├── test-commonsense-v1.5.csv     # KHONG DUNG DUOC: 0% prompt co cau hoi
│   ├── test-anticommonsense-v1.5.csv # KHONG DUNG DUOC: 0% prompt co cau hoi
│   └── test-noncommonsense-v1.5.csv  # KHONG DUNG DUOC: 0% prompt co cau hoi
│
├── src/                       # Thư viện module chức năng
│   ├── cladder.py             # Nạp & nối questions với models qua desc_id (tỷ lệ 100%)
│   ├── perturb.py             # Làm hỏng DAG có kiểm soát: ED, FE, DR (kiểm tra bằng NetworkX)
│   ├── prompts.py             # Bóc DAG văn xuôi, dựng prompt 7 điều kiện, parser đáp án
│   ├── induce.py              # Cho mô hình tự dựng DAG từ văn bản, tính F1 cạnh có phân biệt chiều
│   ├── lexical.py             # Doi ten bien NGAY TRONG cung item:
│   │                          #   KEEP / PERMUTE / SYMBOL / PSEUDO
│   ├── noise.py               # Tiêm 8 loại structured noise vào trước câu hỏi
│   ├── runner.py              # Gọi OpenAI song song 16 luồng, cache đĩa SHA256, tự động retry
│   └── stats.py               # Thống kê: McNemar Exact, Bootstrap tìm k*, mô phỏng Power
│
├── scripts/                   # Kịch bản thực thi thí nghiệm
│   ├── verify_groundtruth.py  # Tự động kiểm chứng nhãn chuẩn CLadder bằng giải tích SCM (0đ)
│   ├── feasibility.py         # Kiểm tra khả thi, độ phân giải mẫu và dự toán ngân sách (0đ)
│   ├── pilot.py               # Chạy thử nghiệm pilot thật (147 items x 6 điều kiện x 3 models)
│   ├── induction.py           # Chạy thí nghiệm Agent tự dựng đồ thị (147 items x 3 models)
│   ├── analyze_lexical.py     # Phan tich thi nghiem tu vung ghep cap (ket qua chinh)
│   ├── analyze_pilot.py       # Phân tích kết quả pilot_raw.csv
│   └── analyze_types.py       # Định giá từng loại lỗi (ED, FE, DR) & kiểm chứng mô hình cộng tính
│
├── results/                   # Lưu trữ toàn bộ 32 bảng CSV kết quả và log chạy thực tế
└── cache/                     # Cache phản hồi API dạng JSON (chạy lại 0đ)
```

---

## PHẦN 4: THIẾT KẾ THỰC NGHIỆM CHI TIẾT (7 Điều kiện Paired)

### 4.1. Bảy điều kiện đối sánh trên cùng một Item
Mỗi câu hỏi được đưa cho mô hình dưới 7 điều kiện prompt khác nhau:

| Điều kiện | Cấu trúc Prompt | Vai trò thực nghiệm |
|---|---|---|
| **PROSE** | Prompt CLadder gốc (DAG nằm lồng trong lời văn) | Đo chi phí của việc bóc cấu trúc ra khỏi văn cảnh |
| **RAW** | Đã bóc sạch câu cấu trúc bằng `strip_structure()` | **ĐƯỜNG SÀN THỰC TẾ (Baseline không có đồ thị)** |
| **ORACLE** | Bóc câu văn xuôi, gắn lại đồ thị đúng thành khối riêng | **TRẦN HIỆU NĂNG (Mức tối ưu khi đồ thị đúng 100%)** |
| **DR_k** | Gắn đồ thị bị **đảo ngược chiều** k cạnh (k=1, 2, 3) | Đo mức độ độc hại của lỗi đảo chiều nhân quả |
| **ED_k** | Gắn đồ thị bị **xóa thiếu** k cạnh (k=1, 2, 3) | Đo mức độ suy giảm do thiếu liên kết |
| **FE_k** | Gắn đồ thị bị **thêm thừa** k cạnh giả (k=1, 2) | Đo mức độ suy giảm do bịa đặt liên kết giả |
| **INDUCED** | Gắn đồ thị **do chính mô hình tự đọc đề bài và tự dựng** | Đo hiệu năng thực tế của AI Agent ngoài đời |

### 4.2. Nhánh đối chứng "Bóc trần năng lực": đổi từ vựng TRONG CÙNG MỘT ITEM

Để kiểm tra xem mô hình thực sự hiểu logic hay chỉ "học vẹt", ta đổi tên biến ngay trên chính item đó và **giữ nguyên đồ thị, các con số, câu hỏi, nhãn chuẩn**:

* **KEEP:** tên biến đời thường như CLadder viết (*poverty, water quality, cholera*).
* **PERMUTE:** **hoán vị chính các tên đó** sang vị trí khác trong đồ thị của chính nó (*cholera has a direct effect on poverty*). Bộ từ vựng giống hệt, độ dài giống hệt (+9 ký tự), chỉ phá chiều nhân quả hợp lẽ thường.
* **SYMBOL:** thay bằng ký hiệu một chữ cái (*A, B, C, D*). Bỏ hẳn từ thật, prompt ngắn đi 169 ký tự.
* **PSEUDO:** thay bằng từ giả của chính CLadder (*rixq, zuph, xevu, glimx*). Thêm việc ký hiệu khó phân biệt.

`PERMUTE` là bộ then chốt: không có nó thì mọi khoảng cách `KEEP` tới `PSEUDO` đều lẫn với việc prompt ngắn đi và tách token khác đi.

Vì cùng item nên **McNemar ghép cặp áp dụng trực tiếp**.

> **Cảnh báo quan trọng:** thiết kế đầu tiên so hai file `test-commonsense-v1.5.csv` và `test-noncommonsense-v1.5.csv`. Cách đó vừa là between-items (0 id trùng nhau), vừa dính lỗi chí mạng: **các file `test-*` không chứa câu hỏi**. Chi tiết ở mục 5.1.

---

## PHẦN 5: KẾT QUẢ THỰC NGHIỆM & CÁC PHÁT HIỆN CỐT LÕI

Dữ liệu chạy thực tế: **19.901 lượt gọi API**, tổng chi phí **19,98 USD**.  
*(Toàn bộ bảng dưới đây được tính trên các câu parse được thành công để tránh thiên vị chống lại RAW).*

---

### 5.1. ĐÍNH CHÍNH TRƯỚC KHI ĐỌC TIẾP: một lỗi nạp dữ liệu đã xoá bỏ "phát hiện chấn động"

Bản trước của tài liệu này đặt phát hiện lớn nhất ở chỗ: *bỏ tên biến có nghĩa thì cả ba model rơi về mức đoán mò 50%, ngay cả khi được đưa đồ thị đúng.*

**Phát biểu đó sai.** Ba file `test-*-v1.5.csv` chỉ chứa bối cảnh và dữ kiện, **không chứa câu hỏi**:

| File | Tỉ lệ prompt có dấu `?` |
|---|:---:|
| `full_v1.5_default.csv` | **100,00%** |
| `test-commonsense-v1.5.csv` | 0,00% |
| `test-anticommonsense-v1.5.csv` | 0,00% |
| `test-noncommonsense-v1.5.csv` | 0,00% |

Cùng một item, cùng story `nonsense`, cùng `query_type=correlation`:

```
full_v1.5_default.csv:  ... The probability of rixq and xevu is 31%.
                        Is the chance of xevu smaller when observing rixq?

test-noncommonsense:    ... The probability of rixq and xevu is 31%.
                        [HET - khong co cau hoi nao]
```

Nhánh pseudoword đã bắt model trả lời yes/no cho prompt **không hỏi gì**. Mức 50% là hệ quả tất yếu của việc buộc đoán, không phải phát hiện về nhận thức. `pilot_raw_noncs.csv` và cột `correct` của `induction_raw_noncs.csv` **không dùng được**.

**Lỗi thứ hai:** nhánh chính lấy mẫu từ `full_v1.5_default.csv`, mà file này **trộn cả ba loại từ vựng** (6.270 dòng story từ thật, 3.842 dòng story `nonsense`). Trong 147 item của nhánh chính có **57 item là từ giả**. Gọi nó là nhánh "commonsense" là sai.

Đã sửa: `make_items` từ chối chạy nếu dưới 99% prompt có dấu `?`, và có thêm cờ `--drop-nonsense`.

---

### 5.2. THIẾT KẾ THAY THẾ: đổi từ vựng ngay trong cùng một item

Cách đo đúng không phải so hai file khác nhau, mà là **đổi tên biến trên chính item đó**, giữ nguyên đồ thị, các con số, câu hỏi và nhãn chuẩn. Module `src/lexical.py` làm việc này bằng `variable_mapping` có sẵn trong metadata CLadder (`background` ánh xạ 1:1 tới nó, 209 background, 0 nhập nhằng).

Ba bộ từ vựng trên **cùng 174 item**, chỉ lấy story từ thật:

| Bộ | Ví dụ | Bỏ đi cái gì |
|---|---|---|
| `KEEP` | *Poverty has a direct effect on water quality and cholera* | không bỏ gì |
| `SYMBOL` | *B has a direct effect on A and D* | tri thức đời thường, giữ ký hiệu dễ phân biệt |
| `PSEUDO` | *Glimx has a direct effect on muvq and xyfo* | tri thức đời thường **và** ký hiệu dễ phân biệt |

Kiểm chứng: **0/174 item còn sót từ vựng gốc, 174/174 giữ đúng số cạnh và số nút.**

Vì cùng item nên **McNemar ghép cặp áp dụng trực tiếp** - đúng thứ thiết kế cũ không có.

---

### 5.3. KẾT QUẢ 1: Đồ thị đúng cắt gần nửa tác hại của việc ẩn danh

| Model | Điều kiện | KEEP | PERMUTE | SYMBOL | PSEUDO |
|---|---|:---:|:---:|:---:|:---:|
| **gpt-4.1** | RAW (không đồ thị) | 75,58 | 65,50 | 67,24 | 66,28 |
| **gpt-4.1** | **ORACLE (đồ thị đúng)** | **87,13** | **78,61** | **78,95** | **79,77** |
| **gpt-4.1-mini** | RAW | 77,91 | 64,91 | 70,52 | 64,37 |
| **gpt-4.1-mini** | **ORACLE** | **84,80** | **83,64** | **82,35** | **81,98** |
| **gpt-4.1-nano** | RAW | 79,19 | 71,08 | 64,16 | 67,24 |
| **gpt-4.1-nano** | **ORACLE** | **74,25** | **61,88** | **68,45** | **68,86** |

McNemar ghép cặp. Gộp 3 bộ ẩn danh x 3 model = **9 phép so sánh mỗi điều kiện**:

| Điều kiện | p<0,05 | Hại TB | Hại lớn nhất |
|---|:---:|:---:|:---:|
| **RAW** (không đồ thị) | **8/9** | **-10,73 pp** | -15,12 |
| **ORACLE** (đồ thị đúng) | **3/9** | **-5,67 pp** | -9,15 |
| PROSE | 2/9 | -4,95 pp | -10,53 |
| DR_k1 (đồ thị sai 1 cạnh) | 4/9 | -7,61 pp | -11,63 |

Tách theo từng bộ:

| Bộ | RAW: hại TB | p<0,05 | ORACLE: hại TB | p<0,05 |
|---|:---:|:---:|:---:|:---:|
| `PERMUTE` | -9,89 | 2/3 | -6,61 | 2/3 |
| `SYMBOL` | -10,47 | **3/3** | -5,41 | 1/3 |
| `PSEUDO` | -11,84 | **3/3** | -5,00 | **0/3** |

> Với `PSEUDO` - bộ khắc nghiệt nhất - đồ thị đưa từ **3/3 xuống 0/3** ô có ý nghĩa.

`PERMUTE` là bộ đồ thị cứu kém nhất (2/3 vẫn có ý nghĩa ở ORACLE). Hợp lý về cơ chế: nó không chỉ lấy đi tri thức đúng mà còn cấp một tri thức **sai lệch** đang cạnh tranh trực tiếp với đồ thị. Đây là dạng nhiễu duy nhất trong bốn bộ có tính đối kháng.

---

### 5.4. KẾT QUẢ 2: Bỏ neo từ vựng làm cấu trúc CÓ GIÁ TRỊ HƠN

`Delta_struct = ORACLE - RAW`:

| Model | KEEP | PERMUTE | SYMBOL | PSEUDO |
|---|:---:|:---:|:---:|:---:|
| **gpt-4.1** | +11,55 | +13,12 | +11,71 | **+13,49** |
| **gpt-4.1-mini** | +6,89 | **+18,72** | +11,83 | +17,61 |
| **gpt-4.1-nano** | **-4,94** | **-9,21** | +4,29 | +1,62 |

Lợi ích của cấu trúc **tăng** khi từ vựng bị bóc, ở hai model mạnh. `gpt-4.1-mini` rõ nhất: +6,89 lên +18,72 dưới `PERMUTE`. Đồ thị thay thế cho tri thức nền đã mất.

`gpt-4.1-nano` đi ngược, và ngược mạnh nhất đúng ở `PERMUTE` (-9,21). Model yếu nhất làm **tệ đi** khi được đưa đồ thị đúng, tệ nhất khi nó vừa có tri thức sai lệch vừa có đồ thị đúng - hai nguồn mâu thuẫn mà nó không đủ sức phân xử. Đây là bằng chứng cụ thể cho luận điểm gốc: **cấu trúc không miễn phí, với model yếu nó có thể gây hại**.

---

### 5.5. KẾT QUẢ 3: Đồ thị SAI đắt hơn khi mất neo từ vựng

Giá một cạnh đảo chiều (`ORACLE - DR_k1`):

| Model | KEEP | PERMUTE | SYMBOL | PSEUDO |
|---|:---:|:---:|:---:|:---:|
| **gpt-4.1** | 10,70 | 9,61 | 10,74 | **14,65** |
| **gpt-4.1-mini** | 7,02 | **11,61** | **15,10** | **14,71** |
| **gpt-4.1-nano** | 1,90 | -0,47 | 3,96 | -2,14 |

So với `KEEP` tại điều kiện `DR_k1`: gpt-4.1 **-11,63 pp (p=0,0012)**, mini **-10,24 pp (p=0,0046)**.

> **CƠ CHẾ ĐƯỢC XÁC LẬP:** khi còn tri thức đời thường, model dùng nó thay cho đồ thị, nên đồ thị đúng ít giúp và đồ thị sai ít hại. Bỏ tri thức đó đi, model buộc phải thực sự dùng cấu trúc, nên đồ thị đúng giúp nhiều hơn và đồ thị sai hại nặng hơn.

---

### 5.6. KẾT QUẢ 4: Toàn bộ chi phí trả ở bậc đầu, không phải độ dài hay việc gắn ký hiệu

Bốn bộ từ vựng tạo thành một bậc thang, mỗi bậc bỏ đi đúng một thứ. Gộp 4 điều kiện x 3 model = **12 phép so sánh mỗi bậc**:

| Bậc | Bỏ đi thêm cái gì | Chênh TB | p<0,05 |
|---|---|:---:|:---:|
| `PERMUTE` - `KEEP` | **chiều nhân quả hợp lẽ** (vẫn từ thật, cùng độ dài) | **-7,52 pp** | **5/12** |
| `SYMBOL` - `PERMUTE` | mất hẳn từ thật, prompt ngắn đi 169 ký tự | +0,73 pp | **0/12** |
| `PSEUDO` - `SYMBOL` | ký hiệu khó phân biệt | +0,13 pp | 2/12 |

**Toàn bộ chi phí của việc ẩn danh được trả ở bậc đầu tiên.** Chỉ hoán vị xem tên nào ứng với vị trí nào trong đồ thị - giữ nguyên từng chữ cái, giữ nguyên độ dài - đã mất 7,52 pp. Xoá sạch từ thật sau đó **không mất thêm gì**.

Ba hệ quả:

1. **Loại được confound độ dài prompt.** `PERMUTE` dài hơn `KEEP` 9 ký tự mà mất 7,52 pp; `SYMBOL` ngắn hơn 169 ký tự mà không mất thêm gì.
2. **Loại được gánh nặng gắn ký hiệu.** Nếu vấn đề là giữ bốn chuỗi vô nghĩa giống nhau qua nhiều bước, `PSEUDO` phải tệ hơn `SYMBOL`. Nó không tệ hơn.
3. **Không phải luận cứ từ việc không bác bỏ được.** Cùng thiết kế, cùng n=174, bậc một phát hiện được hiệu ứng 7,52 pp. Vậy bậc hai và bậc ba không phát hiện được gì là bằng chứng về độ lớn hiệu ứng, không phải thiếu sức mạnh thống kê. Bậc một là **đối chứng dương** cho hai bậc còn lại.

Thứ model mất khi bị ẩn danh là **tri thức về chiều nhân quả nào hợp lẽ trong thế giới thật** - không phải từ vựng, không phải độ dài, không phải khả năng theo dõi ký hiệu lạ.

---

### 5.7. KẾT QUẢ 5: Chất lượng đồ thị Agent tự dựng có giảm thật

Nhánh chính chứa 90 item từ thật và 57 item từ giả, **cùng file, cùng khuôn mẫu, prompt đầy đủ**:

| Model | F1 từ thật | F1 từ giả | Chênh | p (Mann-Whitney) |
|---|:---:|:---:|:---:|:---:|
| **gpt-4.1** | 0,633 | 0,491 | +0,142 | **0,0144** |
| **gpt-4.1-mini** | 0,716 | 0,501 | +0,215 | **0,0002** |
| gpt-4.1-nano | 0,535 | 0,490 | +0,046 | 0,3702 |

Hai model mạnh dựng đồ thị kém hẳn khi tên biến là từ giả, và cả ba hội tụ về F1 khoảng 0,49. Cạnh đảo chiều tăng khoảng gấp đôi (gpt-4.1: 0,022 lên 0,053), **không phải 4-10 lần** như bản cũ báo.

> **GHÉP 5.3 VỚI 5.7 - luận điểm của đề tài:**
> **Neo từ vựng giúp model TRÍCH XUẤT đồ thị nhân quả, chứ không giúp nó SUY LUẬN trên đồ thị đã có.**
>
> *Lưu ý về mức độ tin cậy: nửa "suy luận" (5.3) là ghép cặp nên McNemar áp dụng trực tiếp. Nửa "trích xuất" (5.7) là so sánh between-items 90 với 57 item trong cùng nhánh, nên yếu hơn. Muốn khoá chặt thì phải chạy `induction.py` trên cả bốn bộ từ vựng ghép cặp.*

---

### 5.8. KẾT QUẢ 6: Định giá từng loại lỗi đồ thị - CHƯA XÁC LẬP ĐƯỢC

Đây là câu hỏi nghiên cứu gốc. Bản cũ báo giá tới hai chữ số thập phân. Kiểm lại bằng bootstrap 600 lần, bốc lại theo item:

| Model | Loại | Giá pp/cạnh | CI 95% | R2 | Hoà vốn k\* |
|---|---|:---:|:---:|:---:|:---:|
| gpt-4.1-nano | ED | 2,32 | **[-0,16 ; 4,05]** | 0,72 | - |
| gpt-4.1-nano | FE | 0,58 | **[-7,53 ; 2,05]** | 0,50 | - |
| gpt-4.1-nano | DR | 2,32 | **[-0,32 ; 4,96]** | 0,86 | - |
| gpt-4.1-mini | ED | 2,27 | **[-0,00 ; 4,44]** | 0,79 | - |
| gpt-4.1-mini | FE | 1,48 | **[-1,33 ; 6,67]** | 0,51 | - |
| gpt-4.1-mini | DR | 3,35 | [1,39 ; 7,13] | 0,83 | 0,88 |
| **gpt-4.1** | ED | 2,97 | [0,60 ; 5,30] | 0,68 | 2,56 |
| gpt-4.1 | FE | 2,09 | **[-5,19 ; 2,60]** | 0,19 | - |
| **gpt-4.1** | **DR** | **4,20** | **[1,79 ; 6,57]** | 0,88 | **1,54** |

**6/9 ô có CI độ dốc còn chứa 0.** Không giá `FE` nào được xác lập ở bất kỳ model nào. Thứ bậc `DR > ED > FE` mà bản cũ phát biểu như một quy luật **không đứng vững**.

Chỉ một điểm hoà vốn có CI không chứa 0: `gpt-4.1` với lỗi đảo chiều, **k\* = 1,54, CI [0,13 ; 3,40]**.

Hai lỗi đã sửa trong khâu này:

1. `slope_of` fit đường thẳng có hệ số chặn tự do, nhưng công thức hoà vốn lại giả định đường thẳng xuất phát tại `ORACLE`. Hai đường khác nhau trong một công thức. Riêng ở `nano` với lỗi `DR`, việc này làm k\* nhảy từ 0,51 lên 0,96.
2. Không có khoảng tin cậy nào cả.

**Hệ quả:** khẳng định cũ *"mô hình cộng tính dự đoán chi phí Agent sai lệch dưới 0,3 pp"* cũng bị rút - nó dựa trên các mức giá nay không qua nổi kiểm tra CI.

---

## PHẦN 6: SỔ TAY XỬ LÝ 8 CÁI BẪY KỸ THUẬT KINH ĐIỂN

> Hai bẫy đầu là hai bẫy **đắt nhất**: chúng không làm chương trình chạy sai, chúng làm ra một con số đẹp và sai. Đó là loại lỗi nguy hiểm nhất trong nghiên cứu thực nghiệm.

**0a. Bẫy file dữ liệu thiếu câu hỏi.** Ba file `test-*-v1.5.csv` của CLadder chỉ có bối cảnh và dữ kiện, **0,00% prompt có dấu `?`**, trong khi `full_v1.5_default.csv` là 100%. Chạy trên chúng nghĩa là bắt model trả lời yes/no cho một prompt không hỏi gì, và kết quả là **đúng 50%** - trông y hệt một phát hiện chấn động về nhận thức của LLM. Đây là lỗi đã sinh ra "phát hiện lớn nhất" của bản trước, và nó tồn tại nhiều ngày. **Bài học: trước khi tin bất kỳ con số nào, hãy in ra một prompt hoàn chỉnh và tự đọc.** Đã sửa: `make_items` từ chối chạy nếu dưới 99% prompt có dấu `?`.

**0b. Bẫy file mặc định trộn lẫn split.** `full_v1.5_default.csv` không phải là split commonsense - nó trộn 6.270 dòng story từ thật với 3.842 dòng story `nonsense`. Nhánh chính vì thế có **57/147 item là từ giả** trong khi cả hai tài liệu đều gọi nó là nhánh "commonsense". **Bài học: kiểm tra thành phần thực tế của mẫu, đừng tin tên file.** Đã sửa: cờ `--drop-nonsense`.

1. **Bẫy prompt gốc chứa sẵn DAG:** Prompt CLadder vốn mở đầu bằng *"Husband has a direct effect on wife..."*. Nếu để nguyên thì RAW không phải là baseline không đồ thị. Đã viết `strip_structure()` bóc sạch 100% (10.112/10.112 câu).
2. **Bẫy không gian tên biến (X, Y vs husband, wife):** Ban đầu khối ORACLE ghi *"V2 has a direct effect on Y"* khiến mô hình không map được với bài toán đời thường, làm Delta_struct bị âm ở 2/3 mô hình. Đã sửa lại bằng cách rebuild đồ thị trên chính tên biến của câu chuyện.
3. **Bẫy chữ hoa/thường tạo node ma:** `'Husband'` (đầu câu) và `'husband'` (tân ngữ) bị coi là 2 node khác nhau. Đã chuẩn hóa toàn bộ về chữ thường.
4. **Bẫy chấm câu không parse được thành sai:** Trước đây coi unparsed = sai khiến RAW bị dìm điểm oan (vì RAW parse tốt nhất), tạo ra kết luận sai lệch là "nano không hưởng lợi từ cấu trúc". Khi chỉ tính trên câu parse được: Delta_struct tăng đơn điệu theo tier (+2.22 pp với nano, +4.79 pp với mini, +7.29 pp với gpt-4.1).
5. **Bẫy RNG không ổn định:** Dùng một seed ngẫu nhiên chung khiến việc thêm điều kiện ED/FE làm xáo trộn lại toàn bộ các mẫu DR (gây phương sai tới 3.4 pp, biến p=0.011 thành p=0.150). Đã sửa bằng cách cố định seed độc lập cho từng `(item, loại, k)`.
6. **Bẫy console encoding Windows:** Python trên Windows mặc định dùng cp1252, in ký tự trừ '−' (U+2212) trong CLadder sẽ crash UnicodeEncodeError. Phải luôn thiết lập `PYTHONIOENCODING=utf-8`.

---

## PHẦN 7: HƯỚNG DẪN TÁI LẬP THỰC NGHIỆM TỪ A ĐẾN Z

```bash
# 1. Cài đặt thư viện phụ thuộc
pip install numpy pandas scipy networkx statsmodels openai

# 2. Thiết lập encoding chuẩn (Bắt buộc trên Windows)
export PYTHONIOENCODING=utf-8        # trên Linux / Git Bash
# hoặc: $env:PYTHONIOENCODING="utf-8" # trên PowerShell
# hoặc: set PYTHONIOENCODING=utf-8    # trên CMD

# 3. Tạo file .env chứa OpenAI API Key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 4. Chạy kiểm chứng toán học Ground Truth (0đ, không cần API Key)
python scripts/verify_groundtruth.py

# 5. Chạy phân tích khả thi & Ngân sách (0đ)
python scripts/feasibility.py

# 6. THI NGHIEM TU VUNG GHEP CAP - ket qua chinh cua de tai
#    Cung 174 item, chi doi ten bien. --drop-nonsense loai story tu gia khoi mau
#    goc, vi full_v1.5_default.csv von da tron 38% item tu gia.
for LEX in KEEP PERMUTE SYMBOL PSEUDO; do
  python scripts/pilot.py --n 200 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
      --kmax 1 --types DR --drop-nonsense --lexicon $LEX --tag "_lex$LEX"
done
python scripts/analyze_lexical.py

# 7. Dinh gia tung loai loi do thi (cau hoi nghien cuu goc)
python scripts/pilot.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
                        --kmax 3 --types DR,ED,FE
python scripts/induction.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1
python scripts/analyze_types.py
```

> **KHONG chay tren cac file `test-*-v1.5.csv`.** Chung khong co cau hoi (0% prompt
> co dau `?`) nen moi ket qua se la 50% doan mo. `make_items` gio tu choi chay tren
> chung va bao loi ro rang. Doi tu vung bang `--lexicon`, dung bang cach doi file.

*Lưu ý:* Mọi lượt gọi API đều được lưu đệm trong thư mục `cache/`. Nếu bị rớt mạng hoặc dừng đột ngột, chạy lại lệnh sẽ tiếp tục từ điểm dừng mà **không mất thêm một xu nào**.

---

## PHẦN 8: BÀI HỌC VỀ CƠ CHẾ SELF-GATING & LỘ TRÌNH TIẾP THEO

### 8.1. Vì sao cơ chế Self-Gating cũ không khả thi?

Ban đầu, Giai đoạn 3 dự định xây dựng cơ chế: Agent tự đo điểm chất lượng đồ thị (F1) -> nếu F1 cao thì dùng đồ thị, nếu F1 thấp thì bỏ qua.

Số liệu thực tế chỉ ra `corr(F1, correct)` xấp xỉ 0 trên nhánh chính. Nhưng **phép kiểm này không đủ sức mạnh**: trên văn cảnh từ thật, model gần như không bao giờ đảo chiều cạnh (0,011 tới 0,044 mỗi item), nên phương sai của F1 rất hẹp và tương quan gần 0 là điều được kỳ vọng ngay cả khi có quan hệ thật.

Kết quả mục 5.5 giờ cho biết chỗ để kiểm lại: **dưới `PSEUDO`, một cạnh đảo chiều đắt 14,65 pp thay vì 10,70 pp**, và lỗi đảo chiều của agent tăng gấp đôi. Đó là chế độ mà cổng theo chất lượng đồ thị mới có cơ hội sinh lãi.

**Thiết kế thay thế:** cổng theo **nguy cơ nhầm chiều**, không phải theo mức F1 tổng thể. Agent tự hỏi *"trong bối cảnh này mình có đang đảo ngược nguyên nhân - kết quả không"*, và mục 5.3 tới 5.7 chỉ ra đúng lúc câu hỏi đó đáng giá: khi thiếu neo ngữ nghĩa.

### 8.2. Bốn việc ưu tiên để chốt bài báo hội nghị

1. **Chạy `ED` và `FE` trong thí nghiệm từ vựng.** Hiện chỉ có `DR_k1`. Đây là việc rẻ nhất và trực tiếp nhất để biến mục 5.5 thành một bảng định giá đầy đủ theo từng chế độ từ vựng.
2. **Nâng cỡ mẫu cho phần định giá lỗi đồ thị.** 6/9 ô còn CI chứa 0 (mục 5.8). Đây là câu hỏi nghiên cứu gốc và nó vẫn chưa có câu trả lời.
3. **Chạy induction trên cả ba bộ từ vựng ghép cặp.** Mục 5.7 hiện là so sánh between-items trong cùng nhánh; làm ghép cặp sẽ cho McNemar và khoá chặt luận điểm "neo từ vựng giúp TRÍCH XUẤT chứ không giúp SUY LUẬN".
4. **Thêm ít nhất một dòng model khác họ.** Cả ba model đều là GPT-4.1, một nhà cung cấp, một thế hệ. Caliper dùng 9 model từ 3,8B tới 671B.

### 8.3. Định vị so với Caliper (arXiv:2606.04915)

Caliper đã công bố tháng 6/2026: ẩn danh tên biến làm tụt 7,6 tới 29,6 pp, và khoảng cách sụp khoảng 19 lần trên tập pseudoword của CLadder. Mức tụt 10 tới 13 pp đo được ở đây **nằm trong khoảng đó**, tức là dự án tái lập được Caliper bằng một thiết kế độc lập.

Ba thứ Caliper **không** có, và là đóng góp riêng của dự án:

1. **Điều kiện cấp đồ thị đúng.** Caliper chỉ yêu cầu model tự suy ra cấu trúc, không bao giờ cấp cho nó. Kết quả mục 5.3: cấp đồ thị đúng cắt hơn nửa tác hại của việc ẩn danh.
2. **Chấm chất lượng đồ thị tự dựng, có phân biệt chiều cạnh.** Caliper có prompt scaffold nhưng không đối chiếu cạnh với đáp án (mục 5.7).
3. **Phân rã chi phí theo từng loại lỗi đồ thị** (mục 5.5 và 5.8).

Caliper là **tiền đề**, không phải đối thủ. Cách định vị cũ - *"chúng ta vượt xa họ vì chứng minh model vẫn rơi về đoán mò"* - đã bị rút cùng với lỗi ở mục 5.1.
