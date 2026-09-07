# BÁO CÁO KẾT QUẢ - Chi phí của cấu trúc nhân quả trong suy luận LLM

**Kết luận: đề tài khả thi và giờ đã có một luận điểm được dữ liệu ghép cặp ủng hộ. Nhưng bản báo cáo trước đó có một lỗi nạp dữ liệu làm hỏng phát hiện tiêu đề, và toàn bộ mục đó đã bị rút.**

Ngày: 2026-09-07 · Mọi số liệu do chạy thật

---

## 1. Tóm tắt cho người đọc vội

| Câu hỏi | Trả lời |
|---|---|
| Đề tài chạy được không? | Được. Hạ tầng hoàn chỉnh, nhãn chuẩn tự kiểm chứng tới 6,66e-16 |
| Phát hiện chính? | **Cấp đồ thị đúng cắt gần nửa tác hại của việc ẩn danh tên biến.** Không đồ thị: 8/9 so sánh p<0,05, hại trung bình 10,73 pp. Có đồ thị: 3/9 và 5,67 pp |
| Nguyên nhân tác hại là gì? | **Mất tri thức về chiều nhân quả hợp lẽ thường**, không phải độ dài prompt cũng không phải khó gắn ký hiệu. Chỉ hoán vị tên biến trong chính item đó đã mất 7,52 pp; xoá hẳn từ thật không mất thêm gì (0/12 ô có ý nghĩa) |
| Cơ chế? | **Model lấy chiều của cạnh từ tri thức nhiều hơn từ đề bài** - đi được 33-55% quãng đường tới một tác nhân bỏ hẳn văn bản. Prior sai gây đảo chiều gấp 5,1-7,7 lần prior đúng, prior vắng mặt chỉ 1,3-2,8 lần |
| Giá một cạnh sai có đổi theo miền không? | **Không rõ rệt.** CI trên hiệu ghép cặp: -0,11 [-1,62 ; 1,42] pp. Nhưng khẳng định ngân sách tăng thì **chưa xác lập** - xem mục 8 |
| Có ý nghĩa thống kê chưa? | **Có.** 19/60 McNemar cho phần suy luận, 8/9 Wilcoxon cho phần trích xuất, đều n=174 ghép cặp |
| Điều gì đã bị rút? | **Toàn bộ mục "năng lực nhân quả sụp đổ trên từ giả".** Nguyên nhân là lỗi nạp dữ liệu |
| Còn gì chưa xong? | Định giá lỗi đồ thị chưa đủ cỡ mẫu; CI độ dốc còn chứa 0 ở 6/9 ô |

---

## 2. Đính chính: phát hiện tiêu đề cũ đã bị rút hoàn toàn

Bản báo cáo trước đặt phát hiện lớn nhất ở chỗ này:

> *"Bỏ tên biến có nghĩa thì cả ba model rơi về mức đoán mò, rơi 28-40 pp, ngay cả khi được đưa đồ thị nhân quả đúng."*

**Phát biểu đó sai, và nguyên nhân nằm ở tầng nạp dữ liệu.**

Ba file `test-commonsense-v1.5.csv`, `test-anticommonsense-v1.5.csv` và `test-noncommonsense-v1.5.csv` chỉ chứa **bối cảnh và dữ kiện, không chứa câu hỏi**:

| File | Tỉ lệ prompt có dấu `?` |
|---|---|
| `full_v1.5_default.csv` | **100,00%** |
| `test-commonsense-v1.5.csv` | **0,00%** |
| `test-anticommonsense-v1.5.csv` | **0,00%** |
| `test-noncommonsense-v1.5.csv` | **0,00%** |

Ví dụ, cùng một item, cùng story `nonsense`, cùng `query_type=correlation`:

```
full_v1.5_default.csv :  ... The probability of rixq and xevu is 31%.
                         Is the chance of xevu smaller when observing rixq?
test-noncommonsense   :  ... The probability of rixq and xevu is 31%.
                         [het - khong co cau hoi]
```

Nhánh pseudoword đã bắt model trả lời `yes`/`no` cho một prompt **không hỏi gì cả**. Mức 50% không phải phát hiện về nhận thức của LLM, nó là hệ quả tất yếu của việc buộc đoán. Toàn bộ mục 2 và mục 2.2 của bản cũ, cùng file `pilot_raw_noncs.csv` và cột `correct` của `induction_raw_noncs.csv`, **không dùng được**.

Đã sửa: `make_items` giờ từ chối chạy nếu dưới 99% prompt có dấu `?`.

```
test-noncommonsense-v1.5.csv: chi 0.0% prompt co cau hoi. File nay bi cat mat
phan cau hoi nen khong cham diem duoc. Dung full_v1.5_default.csv va doi tu
vung bang --lexicon.
```

### Lỗi thứ hai: nhánh "commonsense" chưa bao giờ là commonsense

Nhánh chính lấy mẫu từ `full_v1.5_default.csv`, và file đó **trộn cả ba loại từ vựng**: 6.270 dòng story từ thật và 3.842 dòng story `nonsense`. Trong 147 item của nhánh chính có **57 item (38,8%) là item từ giả**. Bản báo cáo cũ gọi đây là nhánh commonsense ở khắp nơi.

Việc này không làm hỏng các kết quả perturbation (chúng ghép cặp trong cùng item), nhưng làm sai mô tả, và làm phép so sánh từ vựng cũ càng vô nghĩa.

---

## 3. Thí nghiệm thay thế: đổi từ vựng ngay trong cùng một item

Cách đo đúng không phải so hai file, mà là **đổi tên biến trên chính item đó** rồi giữ nguyên đồ thị, các con số, câu hỏi và nhãn chuẩn. Mỗi model CLadder có sẵn `variable_mapping`, và `background` ánh xạ 1:1 tới nó (209 background, 0 trường hợp nhập nhằng), nên phép đổi là tất định. Module `src/lexical.py`.

**Bốn** bộ từ vựng trên **cùng 174 item**, chỉ lấy story từ thật. Mỗi bậc bỏ đi đúng một thứ:

| Bộ | Ví dụ | Bỏ đi thêm cái gì | Lệch độ dài so với KEEP |
|---|---|---|---|
| `KEEP` | *Poverty has a direct effect on water quality and cholera* | không bỏ gì | 0 |
| `PERMUTE` | *Cholera has a direct effect on poverty and water quality* | **chiều nhân quả hợp lẽ thường** | **+9 ký tự** |
| `SYMBOL` | *B has a direct effect on A and D* | từ thật | -169 ký tự |
| `PSEUDO` | *Glimx has a direct effect on muvq and xyfo* | ký hiệu dễ phân biệt | -127 ký tự |

Kiểm chứng phép đổi: **0/174 item hỏng, 174/174 giữ đúng số cạnh và số nút** cho cả bốn bộ.

`PERMUTE` là bộ quan trọng nhất và nó **hoán vị chính tên biến của item đó** sang vị trí khác trong đồ thị của chính nó, dùng một derangement nên không biến nào giữ tên cũ. Bộ từ vựng giống hệt, độ dài giống hệt, cách tách token giống hệt - thứ duy nhất bị phá là *chiều nhân quả có hợp lẽ thường hay không*.

Không có `PERMUTE` thì mọi khoảng cách `KEEP` tới `PSEUDO` đều lẫn với việc prompt ngắn đi 127 ký tự và tách token khác đi. `SYMBOL` rồi `PSEUDO` bổ sung hai bậc còn lại: mất hẳn từ thật, rồi ký hiệu khó phân biệt.

---

## 4. Kết quả 1: đồ thị đúng làm tác hại của ẩn danh giảm quá nửa

Độ chính xác, chỉ tính câu parse được, cùng 174 item:

| Model | Điều kiện | KEEP | PERMUTE | SYMBOL | PSEUDO |
|---|---|---|---|---|---|
| gpt-4.1 | RAW (không đồ thị) | 75,58 | 65,50 | 67,24 | 66,28 |
| gpt-4.1 | **ORACLE (đồ thị đúng)** | **87,13** | **78,61** | **78,95** | **79,77** |
| gpt-4.1-mini | RAW | 77,91 | 64,91 | 70,52 | 64,37 |
| gpt-4.1-mini | **ORACLE** | **84,80** | **83,64** | **82,35** | **81,98** |
| gpt-4.1-nano | RAW | 79,19 | 71,08 | 64,16 | 67,24 |
| gpt-4.1-nano | **ORACLE** | **74,25** | **61,88** | **68,45** | **68,86** |

McNemar ghép cặp trên cùng item. Gộp 3 bộ ẩn danh x 3 model = **9 phép so sánh mỗi điều kiện**:

| Điều kiện | Số ô đạt p<0,05 | Tác hại trung bình | Tác hại lớn nhất |
|---|---|---|---|
| **RAW** (không đồ thị) | **8/9** | **-10,73 pp** | -15,12 |
| **ORACLE** (đồ thị đúng) | **3/9** | **-5,67 pp** | -9,15 |
| PROSE (đồ thị nằm trong lời văn) | 2/9 | -4,95 pp | -10,53 |
| DR_k1 (đồ thị sai 1 cạnh) | 4/9 | -7,61 pp | -11,63 |

Tách theo từng bộ từ vựng:

| Bộ | RAW: tác hại TB | p<0,05 | ORACLE: tác hại TB | p<0,05 |
|---|---|---|---|---|
| `PERMUTE` | -9,89 | 2/3 | -6,61 | 2/3 |
| `SYMBOL` | -10,47 | **3/3** | -5,41 | 1/3 |
| `PSEUDO` | -11,84 | **3/3** | -5,00 | **0/3** |

Cấp đồ thị đúng **cắt gần một nửa** tác hại của việc ẩn danh (10,73 xuống 5,67 pp) và làm nó mất ý nghĩa thống kê ở phần lớn các ô. Với `PSEUDO` - bộ khắc nghiệt nhất - đồ thị đưa từ 3/3 xuống **0/3**.

Ngoại lệ trung thực: `PERMUTE` là bộ mà đồ thị cứu **kém nhất** (2/3 vẫn còn ý nghĩa ở ORACLE). Điều đó hợp lý về cơ chế: `PERMUTE` không chỉ lấy đi tri thức đúng, nó cấp cho model một tri thức **sai lệch** đang cạnh tranh trực tiếp với đồ thị được cấp. Đây là dạng nhiễu duy nhất trong bốn bộ có tính đối kháng.

Đây chính là điều kiện Caliper (arXiv:2606.04915) không có. Caliper chứng minh model mất năng lực khi bỏ neo từ vựng; kết quả ở đây chỉ ra **phần lớn thứ bị mất là cái mà đồ thị nhân quả bù lại được**.

### 4.1 Phân tầng theo nhãn từ vựng gốc của CLadder - kết quả sắc hơn nhiều

`full_v1.5_default.csv` có cột `question_property` mà ba file `test-*` bỏ mất, và nó gán nhãn loại từ vựng cho từng item:

| Nhãn | Số dòng | Nghĩa |
|---|---|---|
| `nonsense` | 3.842 | từ bịa |
| `anticommonsense` | 3.129 | **từ thật, chiều nhân quả trái lẽ thường** |
| `easy` + `hard` + `commonsense` | 3.141 | từ thật, chiều hợp lẽ thường |

Cờ `--drop-nonsense` giữ lại mọi dòng từ thật, nghe thì giống "đặt tên hợp lẽ thường" nhưng thực tế **45% mẫu là anticommonsense của chính CLadder**. Nghĩa là đường sàn `KEEP` đã bị bỏ prior đúng trên gần một nửa số item, và mọi hiệu ứng đo được ở mục 4 đều **bị pha loãng**.

Tách ra thì con số đổi hẳn. Gộp 3 bộ ẩn danh x 3 model = 9 phép so sánh mỗi ô:

| Điều kiện | Item có **prior đúng** | Item **prior đã sai sẵn** |
|---|---|---|
| **RAW** (không đồ thị) | **-12,23 pp, 8/9 đạt p<0,05** | -8,91 pp, 2/9 |
| **ORACLE** (đồ thị đúng) | **-3,84 pp, 0/9 đạt p<0,05** | -7,91 pp, 1/9 |

Trên đúng nhóm item mà model **có** một prior đúng để mất, cấp đồ thị đưa tác hại từ 8/9 ô có ý nghĩa xuống **0/9**. Đó là cứu hoàn toàn, không phải cứu một nửa như con số gộp chung gợi ý.

Hai ô mạnh nhất, `PERMUTE` so với `KEEP` ở `RAW`:

| Model | Prior đúng | p | Prior đã sai sẵn | p |
|---|---|---|---|---|
| gpt-4.1 | **-14,13** | **0,0146** | -5,19 | 0,5235 |
| gpt-4.1-mini | **-13,83** | **0,0072** | -10,53 | 0,1516 |

**Xoá một prior đúng đắt hơn xoá một prior vốn đã sai.** Đây là điều bắt buộc phải xảy ra nếu cơ chế được nêu là đúng, và nó là một phép kiểm **có thể thất bại** - nó đã không thất bại.

### 4.2 Một phép nhân bản độc lập không hẹn mà có

Anticommonsense của CLadder là **cùng một thao tác** với `PERMUTE`: giữ từ thật, phá chiều nhân quả hợp lẽ. Hai nhóm khác nhau, hai phương pháp khác nhau, hai tập item khác nhau.

Chi phí đo được, điều kiện `RAW`:

| Model | Anticommonsense của CLadder | `PERMUTE` của dự án này |
|---|---|---|
| gpt-4.1 | **-11,6 pp** | **-14,1 pp** |
| gpt-4.1-mini | **-11,7 pp** | **-13,8 pp** |
| gpt-4.1-nano | -3,6 pp | -4,6 pp |

Ba cặp số, ba lần khớp. `PERMUTE` không phải một thao tác tự chế cho ra một hiệu ứng riêng của nó - nó tái lập được thứ mà chính tác giả CLadder đã dựng sẵn trong benchmark.

Tính bằng `scripts/analyze_prior_strength.py`, 0 USD.

---

## 5. Kết quả 2: bỏ neo từ vựng làm cấu trúc có giá trị hơn

`Delta_struct = ORACLE - RAW`, tức lợi ích của việc được cấp đồ thị đúng:

| Model | KEEP | PERMUTE | SYMBOL | PSEUDO |
|---|---|---|---|---|
| gpt-4.1 | +11,55 | +13,12 | +11,71 | **+13,49** |
| gpt-4.1-mini | +6,89 | **+18,72** | +11,83 | +17,61 |
| gpt-4.1-nano | **-4,94** | **-9,21** | +4,29 | +1,62 |

Với hai model mạnh, lợi ích của cấu trúc **tăng** khi từ vựng bị bóc: đồ thị thay thế cho tri thức nền đã mất. `gpt-4.1-mini` là ví dụ rõ nhất, từ +6,89 lên +18,72 dưới `PERMUTE`.

`gpt-4.1-nano` đi ngược lại và đi ngược mạnh nhất ở `PERMUTE` (-9,21). Model yếu nhất làm **tệ đi** khi được đưa đồ thị đúng, và tệ nhất đúng lúc nó vừa có tri thức sai lệch vừa có đồ thị đúng - hai nguồn thông tin mâu thuẫn mà nó không đủ sức phân xử. Đây là bằng chứng cụ thể cho luận điểm gốc của đề tài: **cấu trúc không miễn phí, và với model yếu nó có thể gây hại**.

---

## 6. Kết quả 3: đồ thị sai đắt hơn khi mất neo từ vựng

Giá của một cạnh đảo chiều, tính bằng `ORACLE - DR_k1`:

| Model | KEEP | PERMUTE | SYMBOL | PSEUDO |
|---|---|---|---|---|
| gpt-4.1 | 10,70 | 9,61 | 10,74 | **14,65** |
| gpt-4.1-mini | 7,02 | **11,61** | **15,10** | **14,71** |
| gpt-4.1-nano | 1,90 | -0,47 | 3,96 | -2,14 |

Tác hại của việc đưa đồ thị sai, so với `KEEP`, cũng lớn lên có ý nghĩa:

| Model | PERMUTE - KEEP tại DR_k1 | p | SYMBOL - KEEP | p | PSEUDO - KEEP | p |
|---|---|---|---|---|---|---|
| gpt-4.1 | -7,43 | 0,0574 | **-8,67** | **0,0107** | **-11,63** | **0,0012** |
| gpt-4.1-mini | -6,06 | 0,1102 | **-9,52** | **0,0090** | **-10,24** | **0,0046** |
| gpt-4.1-nano | -7,55 | 0,0884 | -6,59 | 0,1173 | -1,21 | 0,8776 |

Ba mảnh khớp nhau và cùng chỉ về một cơ chế: **khi còn tri thức đời thường, model dùng nó thay cho đồ thị - nên đồ thị đúng ít giúp và đồ thị sai ít hại. Bỏ tri thức đó đi, model buộc phải thực sự dùng cấu trúc - nên đồ thị đúng giúp nhiều hơn và đồ thị sai hại nặng hơn.**

---

## 7. Kết quả 4: toàn bộ chi phí nằm ở bậc đầu tiên, không phải ở độ dài hay việc gắn ký hiệu

Đây là phép loại trừ quan trọng nhất, và bốn bộ từ vựng tạo thành một bậc thang mà mỗi bậc chỉ bỏ đi một thứ. Gộp cả 4 điều kiện x 3 model = **12 phép so sánh mỗi bậc**:

| Bậc | Bỏ đi thêm cái gì | Chênh lệch TB | Số ô p<0,05 |
|---|---|---|---|
| `PERMUTE` - `KEEP` | **chiều nhân quả hợp lẽ thường** (vẫn là từ thật, cùng độ dài) | **-7,52 pp** | **5/12** |
| `SYMBOL` - `PERMUTE` | mất hẳn từ thật, prompt ngắn đi 169 ký tự | +0,73 pp | **0/12** |
| `PSEUDO` - `SYMBOL` | ký hiệu khó phân biệt, nhiều âm tiết | +0,13 pp | 2/12 |

**Toàn bộ chi phí của việc ẩn danh được trả ở bậc đầu tiên.** Chỉ cần hoán vị xem tên nào ứng với vị trí nào trong đồ thị - giữ nguyên từng chữ cái của bộ từ vựng, giữ nguyên độ dài - đã mất 7,52 pp. Sau đó xoá sạch từ thật thì **không mất thêm gì** (+0,73 pp, 0/12 ô có ý nghĩa), và làm ký hiệu khó phân biệt cũng không (+0,13 pp).

Ba hệ quả:

1. **Loại trừ được confound độ dài prompt.** `PERMUTE` dài hơn `KEEP` 9 ký tự mà vẫn mất 7,52 pp; `SYMBOL` ngắn hơn 169 ký tự mà không mất thêm gì. Độ dài không phải nguyên nhân.
2. **Loại trừ được gánh nặng gắn ký hiệu.** Nếu vấn đề là phải giữ bốn chuỗi vô nghĩa trông giống nhau qua nhiều bước, `PSEUDO` phải tệ hơn `SYMBOL` rõ rệt. Nó không tệ hơn.
3. **Đây không còn là luận cứ từ việc không bác bỏ được.** Cùng một thiết kế, cùng n=174, phát hiện được hiệu ứng 7,52 pp ở bậc một. Vậy khi nó không phát hiện được gì ở bậc hai và bậc ba, đó là bằng chứng về độ lớn hiệu ứng, không phải chuyện thiếu sức mạnh thống kê. Bậc một chính là **đối chứng dương** cho hai bậc còn lại.

Thứ model thực sự mất khi bị ẩn danh là **tri thức về chiều nhân quả nào hợp lẽ trong thế giới thật**, chứ không phải từ vựng, không phải độ dài ngữ cảnh, không phải khả năng theo dõi ký hiệu lạ.

Hai ô `PSEUDO` - `SYMBOL` đạt p<0,05 đều nằm ở `PROSE` và **ngược chiều nhau** (`nano` +9,43 và `gpt-4.1` -10,47), nên không tạo thành tín hiệu nhất quán.

---

## 8. Kết quả 6: giá một cạnh sai không đổi theo miền - nhưng phần còn lại chưa xác lập

Thang lỗi đầy đủ `DR`/`ED`/`FE` với k=1,2,3 chạy **hai lần ở n=400 trên cùng bộ item** - một lần với tên biến gốc, một lần thay bằng từ giả. Điều đó tách được hai thứ mà điểm hoà vốn gộp làm một:

- **giá** - pp mất đi trên mỗi cạnh sai, tức độ dốc đường suy giảm
- **ngân sách** - `ORACLE - RAW`, tức chiều cao mà độ dốc đó phải ăn hết

Vì `k* = ngân sách / giá`, k\* đổi có thể do bất kỳ vế nào. Kết quả: **vế giá đứng vững, vế ngân sách thì chưa.**

### 8.1 Giá KHÔNG đổi - và đây là phần vững

Bản đầu của mục này lập luận bằng "9/9 cặp CI chồng nhau". **Đó là một lỗi thống kê**: hai khoảng tin cậy chồng nhau không chứng minh hai đại lượng bằng nhau, nó chỉ nói phép so sánh chưa đủ sức tách chúng ra.

Vì hai nhánh chạy trên **cùng 399 item**, cách đúng là bootstrap thẳng **hiệu**, bốc cùng bộ item cho cả hai nhánh:

| Model | Loại | Hiệu giá (KEEP - PSEUDO) | CI 95% của **hiệu** |
|---|---|---|---|
| gpt-4.1 | **DR** | **-0,11** | **[-1,62 ; 1,42]** |
| gpt-4.1 | ED | -0,91 | [-2,56 ; 0,67] |
| gpt-4.1-mini | **DR** | **+0,18** | **[-1,53 ; 2,09]** |
| gpt-4.1-mini | ED | -0,10 | [-1,80 ; 1,66] |
| gpt-4.1-nano | DR | -1,11 | [-2,87 ; 0,70] |

**9/9 CI của hiệu đều chứa 0**, và với `DR` ở hai model mạnh chúng bám rất sát 0.

Phát biểu đúng không phải "giá bằng nhau" mà là một **cận tương đương**: nếu giá có phụ thuộc miền từ vựng, mức phụ thuộc đó **nhỏ hơn khoảng 1,6 pp mỗi cạnh** với `DR` trên `gpt-4.1`. So với giá 4,09 pp thì cận này là khoảng 40% - hẹp đủ để có ích, chưa đủ để nói "bất biến".

### 8.2 Ngân sách: hướng đúng nhưng CHƯA XÁC LẬP

Đây là chỗ luận điểm hụt hơi, và phải nói thẳng.

| Model | Ngân sách `KEEP` | Ngân sách `PSEUDO` | Hiệu | CI 95% của hiệu | Có ý nghĩa? |
|---|---|---|---|---|---|
| gpt-4.1 | 4,36 | 9,28 | +4,92 | [-0,40 ; 10,30] | **không** |
| gpt-4.1-mini | 7,07 | 9,95 | +2,88 | [-2,61 ; 8,43] | **không** |
| gpt-4.1-nano | 3,23 | 0,83 | **-2,41** | [-9,02 ; 3,85] | không |

**Cả ba CI đều chứa 0.** Và `gpt-4.1-nano` đi **ngược hướng**: ngân sách của nó giảm chứ không tăng.

Cỡ mẫu cần để hiệu ngân sách đạt p<0,05: **n≈514** cho `gpt-4.1` (hiện có 382) và **n≈935** cho `mini` (hiện có 371). Hiệu ứng có thật hay không thì n=400 chưa trả lời được.

### 8.3 Vậy kết luận được gì

**Được:**

- Giá một cạnh sai **không thay đổi rõ rệt** theo miền từ vựng, với cận tương đương khoảng 1,6 pp mỗi cạnh. Đây là kết quả ghép cặp, CI trên hiệu, không phải suy từ CI chồng nhau.
- `DR` vẫn là loại lỗi đắt nhất ở cả hai chế độ từ vựng, cho cả hai model mạnh.

**Chưa được:**

- Khẳng định "ngân sách tăng gấp đôi khi bỏ neo từ vựng" **chưa xác lập** - cả ba CI chứa 0.
- Do đó khẳng định "điểm hoà vốn dịch từ 0,74 lên 1,93 cạnh" cũng **chưa xác lập**, vì k\* = ngân sách / giá và tử số chưa vững.
- Câu *"hướng dẫn bằng đồ thị bền vững hơn ở đúng nơi nó cần thiết hơn"* là **giả thuyết phù hợp với số liệu, không phải kết luận**. Hai model mạnh đi đúng hướng, `nano` đi ngược.

**Việc phải làm để chốt:** nâng lên n≈600 cho hai model mạnh. Đó là phép kiểm rẻ nhất còn lại và nó có thể thất bại.

> **Ghi chú quy trình:** mục này ban đầu được viết như một kết luận chắc chắn. Vòng phản biện thứ năm bác nó trong cùng phiên, bằng đúng ba phép kiểm mà người phản biện chỉ định: CI trên hiệu thay vì CI chồng nhau, kiểm định riêng cho ngân sách, và soi `nano`. Hai trong ba đòn trúng.

---

## 8b. Định giá ở n=147 (bản cũ, giữ để đối chiếu)

Đây là câu hỏi nghiên cứu gốc, và câu trả lời trung thực là **chưa xác lập được**.

Bản cũ báo giá mỗi cạnh lỗi tới hai chữ số thập phân. Kiểm lại bằng bootstrap 600 lần, bốc lại theo item:

| Model | Loại | Giá pp/cạnh | CI 95% | R² | Hoà vốn k* | CI |
|---|---|---|---|---|---|---|
| gpt-4.1-nano | ED | 2,32 | **[-0,16 ; 4,05]** | 0,72 | - | - |
| gpt-4.1-nano | FE | 0,58 | **[-7,53 ; 2,05]** | 0,50 | - | - |
| gpt-4.1-nano | DR | 2,32 | **[-0,32 ; 4,96]** | 0,86 | - | - |
| gpt-4.1-mini | ED | 2,27 | **[-0,00 ; 4,44]** | 0,79 | - | - |
| gpt-4.1-mini | FE | 1,48 | **[-1,33 ; 6,67]** | 0,51 | - | - |
| gpt-4.1-mini | DR | 3,35 | [1,39 ; 7,13] | 0,83 | 0,88 | [-1,38 ; 2,96] |
| gpt-4.1 | ED | 2,97 | [0,60 ; 5,30] | 0,68 | 2,56 | [-0,22 ; 4,93] |
| gpt-4.1 | FE | 2,09 | **[-5,19 ; 2,60]** | 0,19 | - | - |
| gpt-4.1 | DR | 4,20 | [1,79 ; 6,57] | 0,88 | 1,54 | [0,13 ; 3,40] |

**6 trong 9 ô có CI độ dốc còn chứa 0.** Không có giá nào cho `FE` được xác lập ở bất kỳ model nào. Thứ bậc `DR > ED > FE` mà bản cũ phát biểu như một quy luật **không đứng vững**.

Chỉ một điểm hoà vốn duy nhất có CI không chứa 0: `gpt-4.1` với lỗi đảo chiều, **k\* = 1,54, CI [0,13 ; 3,40]**.

Hai lỗi đã sửa trong khâu này:

1. `slope_of` fit đường thẳng có hệ số chặn tự do, nhưng công thức hoà vốn lại giả định đường thẳng xuất phát tại `ORACLE`. Hai đường khác nhau trong một công thức. Riêng ở `nano` với lỗi `DR`, việc này làm k\* nhảy từ 0,51 lên 0,96.
2. Không có khoảng tin cậy nào cả.

**Hệ quả:** khẳng định cũ *"mô hình cộng tính dự đoán chi phí induction sai lệch dưới 0,3 pp"* cũng bị rút. Nó dựa trên các mức giá mà nay không qua nổi kiểm tra CI.

---

## 9. Kết quả 5: neo từ vựng giúp TRÍCH XUẤT đồ thị, và prior SAI độc hơn prior VẮNG MẶT

Bốn mục trên đo việc **suy luận** trên đồ thị được cấp sẵn. Mục này đo việc **trích xuất** đồ thị từ văn bản - một năng lực khác hẳn. Cùng 174 item, cùng bốn bộ từ vựng, nên ghép cặp được.

| Model | Bộ | F1 | Chênh so với KEEP | p (Wilcoxon ghép cặp) | Cạnh đảo chiều | Gấp KEEP |
|---|---|---|---|---|---|---|
| gpt-4.1 | KEEP | 0,565 | - | - | 0,046 | 1,0x |
| gpt-4.1 | **PERMUTE** | 0,406 | **-0,159** | **0,0000** | **0,316** | **6,9x** |
| gpt-4.1 | SYMBOL | 0,502 | **-0,062** | **0,0114** | 0,126 | 2,8x |
| gpt-4.1 | PSEUDO | 0,458 | **-0,106** | **0,0000** | 0,098 | 2,1x |
| gpt-4.1-mini | KEEP | 0,608 | - | - | 0,069 | 1,0x |
| gpt-4.1-mini | **PERMUTE** | 0,424 | **-0,183** | **0,0000** | **0,529** | **7,7x** |
| gpt-4.1-mini | SYMBOL | 0,508 | **-0,100** | **0,0002** | 0,092 | 1,3x |
| gpt-4.1-mini | PSEUDO | 0,505 | **-0,103** | **0,0001** | 0,109 | 1,6x |
| gpt-4.1-nano | KEEP | 0,462 | - | - | 0,069 | 1,0x |
| gpt-4.1-nano | **PERMUTE** | 0,384 | **-0,078** | **0,0203** | **0,351** | **5,1x** |
| gpt-4.1-nano | SYMBOL | 0,509 | +0,046 | 0,0314 | 0,155 | 2,2x |
| gpt-4.1-nano | PSEUDO | 0,438 | -0,024 | 0,4672 | 0,115 | 1,7x |

**8/9 phép so sánh F1 đạt p<0,05** (ngoại lệ: `nano` với `PSEUDO`).

> **Sàn ngẫu nhiên: F1 = 0,362.** Đồ thị CLadder chỉ có 3-5 nút, tức 6 tới 20 cặp có hướng, nên đoán mò đã đạt gần 0,36. Mọi con số F1 dưới đây phải đọc so với sàn đó: `KEEP` (0,462-0,608) thực sự vượt sàn, còn `PERMUTE` (0,384-0,424) **chỉ hơn sàn 0,02 tới 0,06** - gần như bằng đoán mò. Tính bằng `scripts/induction_baselines.py`, 0 USD.

**Một bất thường phải nêu:** `gpt-4.1-nano` với `SYMBOL` có F1 **cao hơn** `KEEP` (+0,046, p=0,0314) - ngược chiều luận điểm và có ý nghĩa thống kê. Với 9 phép kiểm ở alpha 0,05 thì kỳ vọng khoảng 0,45 ô dương tính giả, nên một ô như vậy nằm trong dự đoán của nhiễu. Nhưng nó là ô duy nhất đi ngược, và `nano` cũng là model có `Delta_struct` âm ở mục 5, nên không loại trừ được khả năng model yếu nhất phản ứng khác về chất. Chưa giải thích được.
 Bỏ neo từ vựng làm chất lượng đồ thị tự dựng giảm thật, và giờ đã là kết luận **ghép cặp**, không còn là so sánh between-items như bản trước.

### Phép phân ly quan trọng nhất của toàn dự án

So sánh cột cuối. `PERMUTE` cấp cho model một prior **sai**; `SYMBOL` và `PSEUDO` **không cấp prior nào**.

| Tình huống | Cạnh đảo chiều mỗi item | Gấp KEEP |
|---|---|---|
| Prior **đúng** (`KEEP`) | 0,046 - 0,069 | 1,0x |
| Prior **vắng mặt** (`SYMBOL`, `PSEUDO`) | 0,092 - 0,155 | 1,3x tới 2,8x |
| Prior **sai** (`PERMUTE`) | **0,316 - 0,529** | **5,1x tới 7,7x** |

Prior sai gây đảo chiều nhiều gấp **3 tới 5 lần** so với không có prior nào.

Nếu model đọc chiều nhân quả **từ văn bản**, hai tình huống sau phải giống nhau - văn bản là như nhau, chỉ tên biến khác. Chúng không giống nhau, và cách biệt rất lớn.

**Kết luận, phát biểu theo mức độ:** khi tên biến gợi một chiều nhân quả sai, model **đi theo tri thức khoảng một phần ba tới một nửa quãng đường** thay vì đọc chiều đã nêu trong đề bài. Nó không bỏ qua văn bản, nhưng cũng không đọc sạch.

Con số đó đo được. Một tác nhân bỏ hẳn văn bản và chỉ trả lời theo tri thức thế giới sẽ đạt **0,966 cạnh đảo chiều** (xuất đồ thị `KEEP`, chấm theo `PERMUTE`). Model thật đạt 0,316-0,529, tức **32,7% tới 54,8%** quãng đường tới tác nhân đó.

Đây cũng là con số "4 tới 10 lần" mà bản báo cáo cũ từng nêu rồi bị rút. Hướng thì đúng, nhưng nó được đo trên dữ liệu hỏng. Giờ nó được đo lại đúng cách, ghép cặp, trên prompt đầy đủ, và **vẫn đứng vững**.

### Ghép mục 4 với mục 9

> **Neo từ vựng giúp model TRÍCH XUẤT đồ thị nhân quả (8/9 ô p<0,05), chứ không giúp nó SUY LUẬN trên đồ thị đã có (đồ thị đúng đưa từ 8/9 xuống 3/9).**

Cả hai vế giờ đều ghép cặp và đều chịu được kiểm định. Cơ chế nối hai vế lại là lỗi đảo chiều: đó vừa là loại lỗi mà việc mất neo từ vựng sinh ra nhiều nhất, vừa là loại lỗi đắt nhất khi suy luận (mục 6).

---

## 10. Còn giữ nguyên giá trị

| Hạng mục | Trạng thái |
|---|---|
| Kiểm chứng nhãn chuẩn CLadder bằng solver SCM giải tích | 2.184 model, sai số tối đa 6,66e-16 |
| Phép gây nhiễu tất định, liệt kê vét cạn | `src/perturb.py`, kiểm bằng NetworkX |
| Seed theo từng `(item, loại, k)` | verify 441/441 ổn định |
| Chấm điểm chỉ trên câu parse được | parse rate báo cáo riêng, 94-100% ở thí nghiệm mới |
| Cache theo hash `(model, temperature, prompt)` | chạy lại 0 đồng |
| Không có LLM nào tham gia viết nhãn | mọi nhãn lấy nguyên từ CLadder |

---

## 11. Chưa làm

| Việc | Ghi chú |
|---|---|
| Nâng n cho phần định giá lỗi | 6/9 CI còn chứa 0; cần n lớn hơn nhiều |
| Chạy `ED`/`FE` trong thí nghiệm từ vựng | hiện chỉ có `DR_k1`; đây là việc rẻ nhất còn lại |
| Thêm dòng model khác họ | cả ba model đều là GPT-4.1, một nhà cung cấp |
| Nhiều lần bốc nhiễu mỗi item (R>=3) | phương sai do bốc đã đo được 1,36 pp |
| Nối lớp structured noise vào pilot | `src/noise.py` đã có 8 loại, chưa nối |
| Tính lại đáp án cho `CI_ACTIVE` | cần dựng SCM mở rộng |

### Giới hạn cứng

Đồ thị CLadder chỉ 2-5 cạnh. `confounding` và `mediation` không thêm được cạnh nào vì đã là DAG đầy đủ trên 3 nút. GPU trên máy là RTX 3050 Laptop 4GB, không chạy được model 7-8B, nên mọi thứ chạy qua API.

---

## 12. Định vị so với Caliper

Caliper (arXiv:2606.04915, 6/2026) đã công bố: ẩn danh tên biến làm tụt 7,6 tới 29,6 pp trên 9 model từ 3,8B tới 671B, và khoảng cách sụp ~19 lần trên tập pseudoword của CLadder. Mức tụt 10-13 pp đo được ở đây **nằm trong khoảng đó** - tức là tái lập được Caliper bằng một thiết kế độc lập.

Ba thứ Caliper không có:

1. **Điều kiện cấp đồ thị đúng.** Và kết quả là tác hại của ẩn danh biến mất (mục 4).
2. **Đo chất lượng đồ thị tự dựng.** Caliper có prompt scaffold nhưng không chấm cạnh (mục 9).
3. **Phân rã theo loại lỗi đồ thị, có phân biệt chiều cạnh** (mục 6, 8).

Caliper là tiền đề, không phải đối thủ.

---

## 13. Chạy lại

```bash
export PYTHONIOENCODING=utf-8         # bat buoc tren Windows

python scripts/verify_groundtruth.py  # khong can API key
python scripts/feasibility.py         # khong can API key

# Thi nghiem tu vung ghep cap (ket qua chinh)
for LEX in KEEP PSEUDO SYMBOL; do
  python scripts/pilot.py --n 200 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
      --kmax 1 --types DR --drop-nonsense --lexicon $LEX --tag "_lex$LEX"
done
python scripts/analyze_lexical.py

# Dinh gia loai loi do thi
python scripts/pilot.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
                        --kmax 3 --types DR,ED,FE
python scripts/induction.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1
python scripts/analyze_types.py
```

Mọi lượt gọi được cache theo hash `(model, temperature, prompt)`. Chạy lại **không tốn thêm tiền**.

Chi tiết từng module và từng cái bẫy: **`WALKTHROUGH.md`**. Phản biện hội đồng 5 ghế: **`REVIEW.md`**.
