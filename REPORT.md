# BÁO CÁO KẾT QUẢ - Khối cấu trúc bù lại tổn thất do ẩn danh tên biến

**Kết luận sau vòng phản biện thứ 7: hiệu ứng có thật, khoảng +6 pp trên 490 item, và nó chịu được bốn đòn tấn công độc lập. Nhưng con số tiêu đề cũ (+14,35 pp) là ước lượng của mẫu nhỏ nhất, và chữ "đúng" trong "đồ thị đúng" đã bị rút vì một đồ thị SAI cũng đạt ngưỡng.**

Ngày: 2026-09-07, cập nhật 2026-09-16 sau vòng phản biện thứ 7 · Mọi số liệu do chạy thật

---

## 1. Tóm tắt cho người đọc vội

| Câu hỏi | Trả lời |
|---|---|
| Đề tài chạy được không? | Được. Hạ tầng hoàn chỉnh. **Nhãn chuẩn thì KHÔNG sạch** - xem mục 10c |
| Phát hiện chính? | **Cấp một khối cấu trúc làm giảm tác hại của việc ẩn danh tên biến, trên nhóm câu hỏi thực sự cần suy luận nhân quả.** Tương tác ghép cặp gộp ba mẫu: **+5,98 pp, CI 95% [+1,78 ; +10,30], p=0,005, n=490 item** |
| Vì sao không còn là +14,35? | +14,35 tái lập chính xác, nhưng nó là ước lượng của **mẫu khám phá n=86**. Hai mẫu lớn hơn đã nằm trên ổ đĩa mà chưa ai kiểm: n=287 cho +6,78 (p=0,009), n=195 cho +1,72 (**p=0,642**). Gộp lại ra +5,98. Xem mục 4.0 |
| Đồ thị có cần ĐÚNG không? | **Chưa chứng minh được, và đây là lỗ hổng lớn nhất.** Thay `ORACLE` bằng đồ thị đảo chiều một cạnh: **+8,64 pp [+1,81 ; +16,38], p=0,018** - vẫn đạt ngưỡng, làm 60% công việc. Hiệu giữa hai bên **không tách khỏi 0**. Xem mục 4.3 |
| Có phải "xoá sạch tác hại" không? | **Không.** Tách hai vế: đồ thị **nâng** nhánh ẩn danh +9,75 pp nhưng **hạ** nhánh `KEEP` -4,60 pp. **32% hiệu ứng là do làm hại điều kiện vốn đang ổn.** Ở nhánh `KEEP`, 2/3 model tệ đi khi nhận đồ thị. Xem mục 4.4 |
| Có phải chỉ do câu lệnh không? | **Không.** Câu lệnh suy luận nhân quả **không có đồ thị** đóng góp dưới **6,4 pp** (cận tương đương; điểm ước lượng +0,1, CI [-6,4 ; +6,3]). Nhưng lưu ý: điều này loại trừ giả thuyết "lời nhắc", **không** loại trừ giả thuyết "khối văn bản lặp lại tên biến" - xem mục 4.3 |
| Nguyên nhân tác hại từ vựng là gì? | **Mất prior đúng, không phải bị gán prior sai.** Bậc thang 5 bậc: mất prior đúng +17,67 pp [+9,24 ; +26,53]; bị gán thêm prior sai **dưới 8,5 pp** (chưa phân giải); từ thật so với ký hiệu **dưới 6,4 pp**. Khi prior đã mất thì từ có thật hay không không còn quan trọng. Xem mục 7 |
| Cơ chế? | **Prior sai độc hơn prior vắng mặt - CHỈ ở nhiệm vụ TRÍCH XUẤT đồ thị.** Đảo chiều 5,1-7,7 lần so với 1,3-2,8 lần, chữ ký đặc hiệu ba cột, sống cả Bonferroni. Với **độ chính xác suy luận** thì không đúng (cận 8,5 pp). Xem mục 9 |
| Đồ thị vào tới đâu? | Chuỗi suy luận nói lại chiều được cấp **45,0%** số lần, so với sàn **2,3-4,6%** khi không được cấp khối nào. Nhưng **75-82%** số item mà chuỗi nói SAI chiều **vẫn cho đúng đáp án**. Chiều cạnh vào tới văn bản; vào tới phép tính khoảng 1/4 trường hợp. Xem mục 9b |
| Giá một cạnh sai có đổi theo miền không? | Không rõ rệt, -0,11 [-1,62 ; +1,42] pp. **CẢNH BÁO: bảng này chưa có script sinh ra** - xem mục 8.1 |
| Ngân sách có tăng khi bỏ neo từ vựng không? | **Đạt ở 2/3 model** trên mẫu gộp n≈800: gpt-4.1 +4,67 [+0,86 ; +8,37], mini +4,38 [+0,88 ; +8,00], nano chứa 0 và **đổi dấu giữa hai lần chạy**. Đây là **nâng cỡ mẫu**, KHÔNG phải nhân bản. Xem mục 8.2 |
| Có ý nghĩa thống kê chưa? | Có, nhưng đọc kèm ba cảnh báo. (a) `p=0,0005` ở mục 4.0 là **sàn bootstrap**, giá trị thật ≈0,00074 (t-test) hoặc 0,0013 (Wilcoxon). (b) Áp Benjamini-Hochberg cho họ 16 phép kiểm tương tác thì **1/3 model** đạt riêng lẻ, không phải 2/3. (c) Tổng phép kiểm trong báo cáo **vượt 200**, BH mới áp cho một họ |
| Cỡ mẫu thật của con số tiêu đề? | **n=490 item** cho con số gộp. Con số +14,35 chạy trên **86 item**, mỗi ô 72-80 sau lọc parse. **Không phải n=174** |
| Điều gì đã bị rút? | Bốn thứ: mục "năng lực sụp đổ trên từ giả" (lỗi nạp dữ liệu, vòng 3); khẳng định ngân sách gấp đôi và k\* (vòng 5); lập luận "đối chứng dương" cho thang từ vựng (vòng 6); và **chữ "đúng" cùng con số +14,35** (vòng 7) |
| Còn gì chưa xong? | Điều kiện `NAMES_ONLY` để cứu chữ "đúng" (1-2 USD, **có thể thất bại**); script cho mục 8.1 và 4.1; một họ model thứ hai |
| Một họ model thứ hai? | **Chưa.** Gemini free tier khảo sát 2026-09-14: RPD=20 cho hầu hết model. Cần key trả phí |

> **Đọc bảng này thế nào.** Mọi con số "chưa phân giải" đều báo bằng **cận tương đương**, không báo bằng "không có hiệu ứng". Mọi khoảng tin cậy đều bootstrap bốc lại theo **item**, không theo ô. Mọi so sánh đều ghép cặp trong cùng item.

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

## 4. Kết quả 1: một khối cấu trúc làm giảm tác hại của ẩn danh

### 4.0 Phân tầng theo nhóm truy vấn - bắt buộc đọc trước mọi con số khác

Vòng phản biện thứ 6 tìm ra rằng mẫu 174 item trộn **ba loại câu hỏi phản ứng với đồ thị theo ba cách hoàn toàn khác nhau**, và việc gộp chúng vào một trung bình đã che mất phát hiện chính.

| Nhóm | n | % mẫu | Đồ thị làm được gì |
|---|---|---|---|
| `marginal`, `correlation` | 56 | 32,2% | **Không gì cả.** Đây là số học thuần trên các con số đã cho. Định lý Causal Hierarchy nói đại lượng bậc 1 được xác định bởi dữ liệu bậc 1 |
| **`backadj`** | 32 | 18,4% | **Đồ thị chính LÀ đáp án.** Đây là câu hỏi *nhận dạng*: "tập hiệu chỉnh nào đúng". Đáp án phụ thuộc duy nhất vào đồ thị |
| `ate`, `ett`, `nde`, `nie`, `det-counterfactual`, `collider_bias`, `exp_away` | 86 | 49,4% | **Đây mới là phần đáng đo.** Đồ thị là công cụ hỗ trợ suy luận, không phải đáp án |

Phân rã `Delta_struct = ORACLE - RAW`, điều kiện `KEEP`:

| Model | rung-1 số học | **`backadj`** | nhân quả thật | gộp lại |
|---|---|---|---|---|
| gpt-4.1 | +0,10 | **+43,75** | +6,66 | +11,55 |
| gpt-4.1-mini | 0,00 | **+59,38** | **-8,61** | +6,89 |
| gpt-4.1-nano | +1,66 | -3,12 | -10,00 | -4,94 |

Ba điều đọc ra từ bảng này:

1. **Trên rung-1, `Delta_struct` gần 0 ở cả ba model (0,00 tới 1,66).** Tách nhỏ hơn: `marginal` thật sự 0,0 ở cả ba; toàn bộ độ lệch nằm ở `correlation` của `nano`. Đây là một **đối chứng âm**, nhưng nó **không hoàn hảo**: DiD trên nhóm này là **-3,38 pp, CI [-7,36 ; +0,41]**, tức một ô mà hiệu ứng thật bằng 0 vẫn đẻ ra |DiD| cỡ 3,4 pp. Đó là **sàn nhiễu của thiết kế**, và con số gộp +5,98 pp phải đọc so với sàn đó.

> **Sửa ở vòng 7: viện dẫn định lý sai chỗ.** Bản trước dùng Causal Hierarchy Theorem để nói đồ thị **phải** vô dụng ở đây. CHT nói điều khác: đại lượng tầng 2 không được xác định bởi dữ liệu tầng 1. Nó không nói một DAG vô dụng khi tính một đại lượng tầng 1 - DAG cấp ràng buộc độc lập có điều kiện và về nguyên tắc có thể giúp. Lý do thực tế nó không giúp: CLadder luôn cho sẵn **đúng những con số cần dùng**. Đó là sự thật về cách đóng gói prompt, không phải một định lý. Giữ CHT cho chỗ nó thật sự áp dụng: giải thích vì sao `ate`/`ett`/`nde`/`nie` không suy ra được từ dữ liệu quan sát nếu không có cấu trúc. Trước vòng 6 nó không được nêu ở đâu.
2. **`backadj` chi phối con số gộp.** Nó chiếm 18,4% mẫu nhưng đóng khoảng 70% `Delta_struct` của `gpt-4.1`. `strip_structure()` xoá đồ thị, tức xoá luôn đáp án - `gpt-4.1-mini` rơi xuống **28,1%, dưới mức đoán mò**, rồi `ORACLE` trả lại 87,5%. Đó không phải đồ thị giúp model suy luận, đó là ta xoá đáp án rồi đưa lại.
3. **Với `gpt-4.1-mini`, bỏ `backadj` ra thì `Delta_struct` đổi dấu sang -8,61 pp.**

#### Phép kiểm tương tác - luận điểm tiêu đề, kiểm đúng cách

Câu *"đồ thị đúng cắt tác hại của ẩn danh"* là một khẳng định về **tương tác**, không phải hai khẳng định rời. Thiết kế ghép cặp hoàn toàn nên tính thẳng được:

```
DiD_i = [(KEEP - LEX) | RAW] - [(KEEP - LEX) | ORACLE]
```

Đọc nó từ chênh lệch **số ô có ý nghĩa** (8/9 so với 3/9) là lỗi Gelman-Stern - đúng lỗi mà mục 8.1 đã tự gọi tên và tự sửa cho phần định giá. Bootstrap bốc lại theo item, 9 ô:

| Mẫu | DiD gộp | CI 95% | p |
|---|---|---|---|
| Gộp cả mẫu | +4,50 pp | [-0,32 ; +9,40] | 0,070 |
| **Chỉ nhân quả thật** | **+14,35 pp** | **[+6,65 ; +22,80]** | **0,0005** |
| Chỉ `backadj` | -5,90 pp | [-18,06 ; +6,94] | 0,392 |
| Chỉ rung-1 số học | -3,38 pp | [-7,36 ; +0,41] | 0,086 |

Tách theo model trên nhóm nhân quả thật: `nano` **+14,00** [+1,78 ; +26,15], `mini` **+21,53** [+9,71 ; +33,69], `gpt-4.1` +7,53 [-4,15 ; +19,58].

**Trên mẫu gộp, luận điểm tiêu đề không đạt ngưỡng 0,05. Lọc đúng nhóm truy vấn thì nó đạt với biên rộng, độ lớn tăng gấp ba, và 2/3 model đạt riêng lẻ.**

McNemar trong từng nhóm, gộp 3 bộ ẩn danh x 3 model:

| Nhóm | `RAW` thô | `RAW` sau BH | `ORACLE` thô | `ORACLE` sau BH |
|---|---|---|---|---|
| Gộp cả mẫu | 8/9, -10,73 pp | 8/9 | 3/9, -5,67 pp | **0/9** |
| **Chỉ nhân quả thật** | **9/9, -17,92 pp** | **9/9** | **0/9, -2,90 pp** | **0/9** |

Tính bằng `scripts/analyze_querygroup.py` (mặc định: seed 20260907, 4.000 lần bootstrap), 0 USD. Mọi cận CI trong mục này lấy từ đúng lần chạy mặc định đó, nên chạy lại ra y hệt.

> **BẮT BUỘC ĐỌC - sửa ở vòng 7. Con số +14,35 pp là ước lượng của mẫu KHÁM PHÁ, không phải con số tiêu đề.**
>
> Ghế thống kê kiểm nó bằng ba đường độc lập (bootstrap 20.000 lần, t-test ghép cặp t=3,50, Wilcoxon) và **nó tái lập chính xác**. Nó không sai. Nhưng nó chạy trên **86 item**, và dự án đã trả tiền cho hai mẫu lớn hơn đang nằm trên ổ đĩa mà chưa ai kiểm:
>
> | Mẫu | kmax | họ đồ thị | n item nhân quả | DiD | p |
> |---|---|---|---|---|---|
> | n=174 (khám phá) | 1 | 10 | 85 | +12,71 | 0,018 |
> | n=580 | 1 | 10 | 287 | +6,78 | 0,009 |
> | n=399 | 3 | 7 | 195 | +1,72 | **0,642** |
> | **Gộp, bỏ trùng theo id gốc** | | | **490** | **+5,98** | **0,005** |
>
> CI gộp: **[+1,78 ; +10,30]**. Hiệu ứng **có thật** - hai trên ba mẫu đạt, và bốn giả thuyết cạnh tranh đều bị loại trừ (mục 4.7). Nhưng độ lớn thật khoảng **6 pp**, và +14,35 cao gấp 2,4 lần vì được đọc ở mẫu nhỏ nhất. Đó là hình dạng của lời nguyền người thắng cuộc.
>
> Ba mẫu **không độc lập**: trùng 24-35% item, phần trùng dùng lại phản hồi cache y hệt từng byte, và mẫu n=399 dùng kmax=3 nên chỉ phủ 7 họ đồ thị thay vì 10. Không được gọi là nhân bản.
>
> **Ba cảnh báo thống kê phải đi kèm mọi con số ở mục này:**
>
> 1. **`p = 0,0005` là SÀN của bootstrap**, không phải phép đo. Giá trị nhỏ nhất khác 0 mà `2*min(...)` trả được là 2/n = 0,0005. Chạy 20.000 lần vẫn chạm sàn. Giá trị có ý nghĩa: **p = 0,00074** (t-test trên DiD mỗi item) hoặc **0,0013** (Wilcoxon). Nên viết "p < 0,001".
> 2. **"2/3 model đạt riêng lẻ" thành 1/3 nếu hiệu chỉnh.** Họ tương tác gồm 4 nhóm x 4 phạm vi = 16 phép kiểm, hiện **không hiệu chỉnh**, trong khi họ 9 ô McNemar thì có. Áp cùng hàm `bh()`: gộp 9 ô sống (p=0,0003), `mini` sống (p=0,0003), `nano` chết (p=0,0235). Đây là bất nhất nội bộ.
> 3. **Cỡ mẫu thật là 86 item, mỗi ô 72-80 sau lọc parse.** Không phải n=174. Nhóm `backadj` chỉ 32 item.
>
> **Quyết định lọc là HẬU KIỂM.** Cách cắt theo nhóm truy vấn do ghế R2 nêu ở vòng 6, **sau khi** phép kiểm gộp trượt ở p=0,070. Định lý chỉ biện minh cho việc bỏ rung-1; bỏ `backadj` là một lập luận, tốt nhưng là lập luận, và nó đóng góp nhiều hơn:
>
> | Cách cắt | DiD | CI 95% | p |
> |---|---|---|---|
> | Gộp tất cả | +4,50 | [-0,46 ; +9,46] | 0,070 |
> | Chỉ bỏ rung-1 (có định lý đỡ) | +8,37 | [+1,78 ; +15,40] | 0,012 |
> | Chỉ bỏ `backadj` | +7,13 | [+2,01 ; +12,67] | 0,006 |
> | Bỏ cả hai | +14,35 | [+6,77 ; +23,21] | sàn |
>
> **Hai loại truy vấn xếp sai nhóm** (vòng 7): `exp_away` là rung 1 theo nhãn của chính CLadder nhưng đang ở nhóm `causal`; `collider_bias` thì đồ thị **chính là** đáp án, y hệt `backadj`, cũng đang ở `causal`. Sửa cả hai: DiD thành **+14,53 [+5,74 ; +23,45]**. Không lật kết quả, nhưng đó là may - `Delta_struct` của `collider_bias` tình cờ bằng đúng 0,0 ở cả ba model.
>
> **Tiêu chí phân nhóm, thành văn:** *một loại truy vấn thuộc nhóm `identify` nếu đáp án của nó là hàm của riêng đồ thị, độc lập với mọi tham số số học.* Chỉ `backadj` và `collider_bias` thoả.
>
> **`RAW` không phải "không đồ thị".** `strip_structure()` chỉ xoá câu khớp `has a direct effect on`. Với `det-counterfactual` (20/174 item, **23% nhóm nhân quả**), hai câu còn lại chính là phương trình cấu trúc `X -> V2` và `(X, V2) -> Y`. Khớp với dữ liệu: `Delta_struct` của loại này là **-30,0 / -20,0 / -5,0**, âm nặng ở cả ba model. Bỏ nó ra: DiD **+13,85 [+3,58 ; +24,86] p=0,0085**. Bền, nhưng phải công bố.

### 4.0b Bảng gộp cũ, giữ để đối chiếu

Mọi con số dưới đây **gộp cả ba nhóm truy vấn**, nên chúng bị `backadj` pha loãng. Giữ lại để so với các vòng phản biện trước.

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

Cấp đồ thị **cắt gần một nửa** tác hại của việc ẩn danh (10,73 xuống 5,67 pp) và làm nó mất ý nghĩa thống kê ở phần lớn các ô. Với `PSEUDO` - bộ khắc nghiệt nhất - đồ thị đưa từ 3/3 xuống **0/3**.

> **Sửa ở vòng 7.** Cách đọc bằng **đếm ô** ở bảng này là lỗi Gelman-Stern mà chính mục 4.0 tự gọi tên. Và cụm "cắt gần một nửa" chỉ đúng nếu ta không tách hai vế: mục 4.4 cho thấy **32% mức cắt đó là do đồ thị hạ nhánh `KEEP` xuống**, không phải nâng nhánh ẩn danh lên. Bảng này giữ để đối chiếu lịch sử, không phải để trích.

Ngoại lệ trung thực: `PERMUTE` là bộ mà đồ thị cứu **kém nhất** (2/3 vẫn còn ý nghĩa ở ORACLE). Điều đó hợp lý về cơ chế: `PERMUTE` không chỉ lấy đi tri thức đúng, nó cấp cho model một tri thức **sai lệch** đang cạnh tranh trực tiếp với đồ thị được cấp. Đây là dạng nhiễu duy nhất trong bốn bộ có tính đối kháng.

Đây chính là điều kiện Caliper (arXiv:2606.04915) không có. Caliper chứng minh model mất năng lực khi bỏ neo từ vựng; kết quả ở đây chỉ ra **phần lớn thứ bị mất là cái mà đồ thị nhân quả bù lại được**.

### 4.1 Phân tầng theo nhãn từ vựng gốc của CLadder - kết quả sắc hơn nhiều

> **Tư cách của mục này: PHÂN TÍCH HẬU KIỂM, không tiền đăng ký.** Cách phân tầng này xuất hiện ở vòng phản biện thứ 4, sau khi mục 4 đã viết xong và ba vòng phản biện đã chạy trên số chưa phân tầng. Kho không có file tiền đăng ký nào và `scripts/analyze_prior_strength.py` chỉ có đúng một commit - chính commit giới thiệu phát hiện này.
>
> Điều đó không làm nó sai: phân tầng có động cơ cơ chế rõ, cột `question_property` là nhãn của CLadder chứ không phải nhãn tự chế, và ranh giới tầng không có bậc tự do để vặn. Nhưng nó phải được đọc là **khám phá cần nhân bản**, không phải xác nhận. Không gian cách cắt sẵn có gồm ít nhất `question_property`, `rung`, `query_type`, `graph_id`, `story_id` - người đọc không có cách nào biết đã nhìn bao nhiêu cách.

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

#### Hai khẳng định của mục này có số phận khác nhau - kiểm ở vòng 6

**(a) "Đồ thị cứu mạnh hơn hẳn trên nhóm có prior đúng" - ĐƯỢC XÁC LẬP.** Phép kiểm tương tác ghép cặp, bootstrap theo item, chạy riêng trong từng tầng:

| Tầng | Tương tác (DiD) | CI 95% | p |
|---|---|---|---|
| **Prior đúng** | **+7,85 pp** | **[+2,30 ; +13,97]** | **0,003** |
| Prior đã sai sẵn | +0,26 pp | [-7,83 ; +8,86] | 0,941 |

Đây là phân ly sạch, đúng hướng cơ chế dự đoán, và là một phép kiểm **có thể thất bại**. Nó không thất bại.

**(b) "Xoá một prior đúng đắt hơn xoá một prior vốn đã sai" - CHƯA XÁC LẬP.** Bản trước suy khẳng định này từ -12,23 so với -8,91 và 8/9 so với 2/9. Kiểm trực tiếp trên hiệu giữa hai tầng (hai tập item rời nhau nên không ghép cặp được, bootstrap riêng trong từng tầng):

```
hai tren prior DUNG tru hai tren prior SAI SAN CO
  -3,31 pp   CI 95% [-12,63 ; +6,61]   p = 0,503
  3/9 o di nguoc chieu khang dinh
```

Hướng thì đúng, nhưng CI chứa 0. Và phần lớn chênh lệch "8/9 so với 2/9" là **công suất chứ không phải độ lớn**: tầng prior đúng có n trung bình 91-94 item, tầng prior sai chỉ 73-77, nên tầng sau kém nhạy khoảng 15% một cách hệ thống và **bắt buộc** phải đếm ra ít ô hơn ngay cả khi hiệu ứng thật bằng nhau.

**Phát biểu đúng:** *trên nhóm item có prior đúng, cấp đồ thị đưa tác hại từ -12,23 pp xuống -3,84 pp, và phép kiểm tương tác cho +7,85 pp [+2,30 ; +13,97] so với +0,26 pp ở nhóm prior đã sai sẵn. Chênh lệch tác hại giữa hai tầng thì đúng hướng nhưng chưa đạt ngưỡng ở cỡ mẫu này.*

### 4.2 Bằng chứng hội tụ yếu từ nhãn anticommonsense của CLadder

> **Đã hạ cấp ở vòng phản biện thứ 6.** Mục này trước đây mang tiêu đề "một phép nhân bản độc lập không hẹn mà có" và được dùng làm bằng chứng hiệu lực cấu trúc cho toàn bộ thao tác `PERMUTE`. Hai lý do buộc phải hạ giọng, cả hai đều kiểm được:
>
> **Thứ nhất, hai nhánh không độc lập.** Chúng dùng chung **24 trên 37 story**, cùng một lần chạy, cùng một model, cùng điều kiện `KEEP`/`RAW`. "Hai tập item khác nhau" là mô tả sai.
>
> **Thứ hai, và nặng hơn: khoảng cách sụp một nửa khi cân bằng hiệp biến.** Hai nhánh lệch nhau về thành phần `query_type` - `correlation` chiếm 18,1% nhánh prior-đúng so với 9,0% nhánh prior-sai; `backadj` 14,9% so với 23,1%; `exp_away` 4,3% so với 0%. Gán lại trọng số nhánh prior-đúng theo phân bố của nhánh prior-sai:
>
> | Model | Thô | **Sau cân bằng** | `PERMUTE` tương ứng |
> |---|---|---|---|
> | gpt-4.1 | +11,6 | **+6,2** | -14,1 |
> | gpt-4.1-mini | +11,7 | **+6,3** | -13,8 |
> | gpt-4.1-nano | +3,6 | **-0,2** | -4,6 |
>
> "Ba cặp số, ba lần khớp" thành ba cặp lệch hơn hai lần, và cặp thứ ba đổi dấu. **Đây là bằng chứng hội tụ yếu, chưa cân bằng hiệp biến - không phải một phép nhân bản.**
>
> Bảng dưới giữ nguyên số thô để đối chiếu.

Anticommonsense của CLadder là một thao tác **cùng hướng** với `PERMUTE`: giữ từ thật, phá chiều nhân quả hợp lẽ. Hai thao tác khác nhau về cách thực hiện - anticommonsense thay bằng một bộ từ vựng khác vẫn tự nhất quán, còn `PERMUTE` hoán vị chính từ vựng của item nên có thể phá mạch văn.

Chi phí đo được, điều kiện `RAW`:

| Model | Anticommonsense của CLadder | `PERMUTE` của dự án này |
|---|---|---|
| gpt-4.1 | **-11,6 pp** | **-14,1 pp** |
| gpt-4.1-mini | **-11,7 pp** | **-13,8 pp** |
| gpt-4.1-nano | -3,6 pp | -4,6 pp |

Ba cặp số thô cùng hướng. Nhưng sau khi cân bằng theo `query_type` (xem khối cảnh báo ở đầu mục), khoảng cách còn +6,2 / +6,3 / -0,2 pp - tức chỉ khoảng 40% độ lớn của `PERMUTE`, và `nano` mất hẳn tín hiệu.

**Phát biểu đúng:** CLadder gán nhãn anticommonsense cho 45% mẫu này. So sánh giữa-item, chưa cân bằng hiệp biến, cho khoảng cách cùng hướng với `PERMUTE`; sau khi cân bằng thì khoảng cách co lại đáng kể. Đây là bằng chứng **hội tụ**, không phải **nhân bản**.

Tính bằng `scripts/analyze_prior_strength.py`, 0 USD.

---

### 4.3 Đồ thị có cần ĐÚNG không - phép kiểm dự án tự trượt

Đây là lỗ hổng lớn nhất còn lại, và nó do hội đồng vòng 7 tìm ra.

Điều kiện `RAW_INSTR` ở mục 4.5 loại trừ được giả thuyết "chỉ là lời nhắc suy luận". Nhưng nó **không** loại trừ được giả thuyết mạnh hơn: *cái giúp không phải nội dung cấu trúc, mà là sự có mặt của một khối văn bản nhắc lại tên biến kèm vài mũi tên*. `RAW_INSTR` thêm 65 ký tự và **không chứa một tên biến nào**.

Đối chứng đúng đã nằm sẵn trong kho: các khối đồ thị **sai**. Cùng độ dài, cùng câu lệnh, cùng danh sách tên biến, chỉ khác là sai.

| Khối được cấp | DiD | CI 95% | p |
|---|---|---|---|
| `ORACLE` đồ thị **đúng** | +14,31 | [+6,30 ; +22,52] | 0,001 |
| `PROSE` đúng, trong lời văn | +13,91 | [+7,30 ; +20,47] | <0,001 |
| **`DR_k1` đảo chiều một cạnh** | **+8,64** | **[+1,81 ; +16,38]** | **0,018** |

Hiệu ghép cặp `ORACLE` trừ `DR_k1` gộp ba mẫu: **+1,62 pp, CI [-2,07 ; +5,30], p=0,40.** Không tách khỏi 0.

**Ở cỡ mẫu này dự án không phân biệt được đồ thị đúng với đồ thị sai**, ở đúng chỗ luận điểm tiêu đề đứng.

Phản bác "`DR_k1` chỉ sai 1 trên 2-5 cạnh nên vẫn gần đúng" không cứu được:

- `DR_k3` ở n=399 cũng ngang `ORACLE`: -1,48 [-7,81 ; +4,91]
- `results/errortype_by_lexicon.csv` cho thấy dưới `KEEP`, giá của `DR` chỉ 2,48 pp với **CI chứa 0** - trong thiết lập này độ đúng của đồ thị gần như không tính tiền

**Phát biểu còn giữ được:** *cấp một khối cấu trúc - dù đúng hay sai một cạnh - làm giảm tác hại của ẩn danh.* Đây là mệnh đề khác hẳn về cơ chế, và nó nghiêng về **tái gắn ký hiệu** chứ không phải suy luận cấu trúc. Mục 4.6 cho thêm hai bằng chứng cùng hướng.

> **Phép kiểm phục hồi, chưa chạy.** Điều kiện `NAMES_ONLY`: cùng vị trí, cùng câu lệnh, liệt kê **đúng các tên biến, không một mũi tên nào** ("The variables of this world are: A, B, C."). Nếu `ORACLE` vượt `NAMES_ONLY` có ý nghĩa trên nhánh ẩn danh thì mệnh đề "nội dung cấu trúc" sống lại. Nếu không, mệnh đề đúng là về tái gắn ký hiệu. Ước 1-2 USD. **Phép kiểm này có thể thất bại, và đó là lý do nó đáng chạy.**

### 4.4 Nâng nhánh xấu, hay hạ nhánh tốt

"Xoá sạch tác hại" có thể đạt được bằng hai cách rất khác nhau. Tách ra:

| | Đồ thị làm thay đổi |
|---|---|
| Nhánh ẩn danh | **+9,75 pp** nâng lên |
| Nhánh `KEEP` | **-4,60 pp** hạ xuống |
| Tương tác | +14,35 pp |

**32% của hiệu ứng là đồ thị làm hại điều kiện vốn đang ổn.**

Theo model, ở nhánh `KEEP`, nhóm câu hỏi nhân quả:

| Model | `Delta_struct` ở `KEEP` |
|---|---|
| gpt-4.1-nano | **-12,39** |
| gpt-4.1-mini | **-7,68** |
| gpt-4.1 | +6,27 |

**Hai trên ba model làm tệ đi khi được cấp đồ thị**, trên chính nhóm câu hỏi làm nên luận điểm.

Điều này gần với chính caveat Caliper báo về scaffold của họ (mục 4.9 của họ): *"thu hẹp khoảng cách chủ yếu bằng cách hạ P0 chứ không phải phục hồi P1"*. Dự án **tái lập một phần caveat đó** chứ không thoát khỏi nó.

**Hệ quả cho cách phát biểu.** Câu *"model dùng được cấu trúc nhân quả khi được đưa cho"* chỉ đúng ở mức **tương tác**. Ở mức tuyệt đối nó sai với 2/3 model. Mọi bảng DiD từ nay phải in `dKEEP` và `dANON` riêng.

### 4.5 Câu lệnh hay nội dung đồ thị

`ORACLE` cộng vào hai thứ cùng lúc: khối nội dung đồ thị, và câu "Use this causal structure when reasoning". Điều kiện `RAW_INSTR` mang câu lệnh mà không có đồ thị.

| Phép kiểm | DiD | CI 95% | p |
|---|---|---|---|
| A. `RAW` so với `ORACLE` | +12,71 | [+2,11 ; +23,11] | 0,019 |
| B. **chỉ câu lệnh** | +0,10 | **[-6,40 ; +6,34]** | 1,000 |
| C. **đồ thị, câu lệnh đã ở trong nền** | **+13,48** | [+4,27 ; +22,95] | 0,003 |

**Phát biểu đúng cho B là một cận tương đương: ở cỡ mẫu này, phần đóng góp của câu lệnh nhỏ hơn khoảng 6,4 pp.** Không được viết "+0,00 pp" và đọc như một kết quả null - CI tương thích với việc câu lệnh tái tạo tới một nửa tương tác.

Ở **mức tuyệt đối** thì khác hẳn: trên nhánh `KEEP`, câu lệnh chiếm phần lớn `Delta_struct` của `gpt-4.1` (+3,53 trên +6,66). Với `mini` tổng là âm nên phân rã không áp dụng; với `nano` câu lệnh **hại** 7,16 pp.

**Hệ quả:** con số "một đồ thị đáng bao nhiêu điểm" ở mức tuyệt đối **không dùng được**, vì nó không ổn định giữa các model và lẫn với hiệu ứng câu lệnh.

> **Giới hạn của phép tách này.** Cách diễn đạt không thể trùng từng byte: `ORACLE` nói "this causal structure", `RAW_INSTR` phải nói "the causal structure of this world" vì không có "this" khi không có khối. Và như mục 4.3 chỉ ra, `RAW_INSTR` loại trừ giả thuyết "lời nhắc" nhưng **không** loại trừ giả thuyết "khối có tên biến".

### 4.6 Hiệu ứng tập trung ở đâu

Hai biến điều tiết lớn, cả hai đều nghiêng về giả thuyết tái gắn ký hiệu.

| Phạm vi | n | DiD | CI 95% | p |
|---|---|---|---|---|
| Đồ thị **3 nút** | 40 | +5,72 | [-3,26 ; +15,98] | 0,217 |
| Đồ thị **>= 4 nút** | 46 | **+22,44** | [+9,55 ; +35,40] | 0,0005 |
| Truy vấn `ate` | 24 | **+27,44** | [+12,74 ; +42,97] | 0,0005 |
| Truy vấn `det-counterfactual` | 20 | +16,29 | [+1,26 ; +31,92] | 0,033 |
| Truy vấn `ett` | 21 | +2,48 | [-11,02 ; +17,39] | 0,760 |
| Truy vấn `nie` | 11 | +6,43 | [-19,73 ; +36,03] | 0,655 |

Tương quan DiD với số nút: rho=+0,121, p=0,0013 (n=699 quan sát ô-item). Bỏ `ate` ra khỏi nhóm: +9,59 [+0,88 ; +19,21], chạm mép.

**Hai câu phải nói ra.** Thứ nhất, *"trên nhóm câu hỏi thực sự cần suy luận nhân quả"* thực chất là **"chủ yếu trên câu hỏi ATE"**. `ate` và `ett` là hai truy vấn gần nhau nhất về hình thức, và chúng cho +27,44 với +2,48. Thứ hai, **nửa mẫu - các đồ thị 3 nút - không cho kết quả nào.**

Càng nhiều ký hiệu phải giữ thì khối đồ thị càng có ích. Đó là dấu hiệu của trí nhớ làm việc, không phải của suy luận cấu trúc.

### 4.7 Bốn giả thuyết cạnh tranh đã loại trừ được

Phần này quan trọng ngang phần bị rút. Mỗi mục đóng hẳn một hướng phản biện.

| Giả thuyết | Kiểm thế nào | Kết quả |
|---|---|---|
| **Độ dài prompt** | Hồi quy `correct` theo `in_tok`, 24 ô | **Loại trừ, và loại trừ có lợi.** r trung bình -0,113; **8 ô âm có ý nghĩa, 0 ô dương**. Prompt dài hơn làm TỆ hơn. Thêm nữa khối `ORACLE` thêm 221 ký tự trên `KEEP` nhưng chỉ 157-175 trên bộ ẩn danh, nên lý thuyết độ dài dự đoán DiD **âm**; quan sát được là dương |
| **Residue từ thật còn sót** | Tách item sạch và item còn sót | **Loại trừ.** Sạch +16,15 [-4,99 ; +39,23]; còn residue +13,71 [+5,49 ; +22,10]. Residue **làm co** hiệu ứng chứ không tạo ra nó |
| **Một lát cắt may mắn** | Bốc ngẫu nhiên 86/174 item, 2.000 lần | **Loại trừ.** Trung bình +4,43, độ lệch 2,59, và **0/2.000** lần chạm tới +14,35 |
| **Độ nhạy chấm điểm** | Chấm câu không parse được thành sai | **Loại trừ.** +14,35 thành +14,60. Parse rate thấp nhất 93,0% |

Đòn "chuỗi suy luận chỉ sao chép khối đã cấp" cũng loại trừ được: cắt mọi câu trùng lặp với khối, tỉ lệ "theo đồ thị" chỉ giảm 1,2 tới 10,3%.

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

| Bậc | Bỏ đi thêm cái gì | Chênh lệch TB | **CI 95% bootstrap theo item** |
|---|---|---|---|
| `PERMUTE` - `KEEP` | **chiều nhân quả hợp lẽ thường** (vẫn là từ thật, cùng độ dài) | **-7,53 pp** | **[-11,41 ; -3,81]** |
| `SYMBOL` - `PERMUTE` | mất hẳn từ thật, prompt ngắn đi 169 ký tự | +0,73 pp | **[-2,43 ; +3,74]** |
| `PSEUDO` - `SYMBOL` | ký hiệu khó phân biệt, nhiều âm tiết | +0,13 pp | **[-1,85 ; +2,05]** |

> **Đổi cách trình bày ở vòng phản biện thứ 6.** Bảng này trước đây báo "số ô p<0,05" (5/12, 0/12, 2/12). Đếm ô **không tách được "bằng 0" khỏi "thiếu công suất"**, và mười hai ô đó dùng chung 174 item nên chúng không phải mười hai quan sát trao đổi được. CI bootstrap bốc lại theo item thì tách được, và nó **có thể sai**.
>
> Đọc bảng mới: bậc 2 loại trừ mọi tác hại lớn hơn **2,4 pp**, bậc 3 loại trừ mọi tác hại lớn hơn **1,9 pp**, so với **7,53 pp** ở bậc 1. Đó là cận tương đương thật, mạnh hơn hẳn "0/12 ô".

**Toàn bộ chi phí của việc ẩn danh được trả ở bậc đầu tiên.** Chỉ cần hoán vị xem tên nào ứng với vị trí nào trong đồ thị - giữ nguyên từng chữ cái của bộ từ vựng, giữ nguyên độ dài - đã mất 7,52 pp. Sau đó xoá sạch từ thật thì **không mất thêm gì** (+0,73 pp, 0/12 ô có ý nghĩa), và làm ký hiệu khó phân biệt cũng không (+0,13 pp).

Ba hệ quả:

1. **Loại trừ được confound độ dài prompt.** `PERMUTE` dài hơn `KEEP` 9 ký tự mà vẫn mất 7,52 pp; `SYMBOL` ngắn hơn 169 ký tự mà không mất thêm gì. Độ dài không phải nguyên nhân.
2. **Loại trừ được gánh nặng gắn ký hiệu.** Nếu vấn đề là phải giữ bốn chuỗi vô nghĩa trông giống nhau qua nhiều bước, `PSEUDO` phải tệ hơn `SYMBOL` rõ rệt. Nó không tệ hơn.
3. **Đây không còn là luận cứ từ việc không bác bỏ được** - nhưng lý do không phải cái đã viết trước đây.

> **Lập luận "đối chứng dương" đã bị rút ở vòng 6.** Bản trước lập luận: *"cùng thiết kế, cùng n=174, phát hiện được 7,52 pp ở bậc một, vậy việc không phát hiện gì ở bậc hai là bằng chứng về độ lớn hiệu ứng"*. Lập luận đó **không hợp lệ ở mức ô lẻ**. Công suất của McNemar phụ thuộc số cặp bất đồng, và ở bậc 1 công suất tại chính độ lớn hiệu ứng của nó chỉ khoảng 0,44 - đúng như vậy nó chỉ bắt được 5/12 ô. Một đối chứng dương chạy ở khoảng 50% công suất không cho phép suy ra điều gì về bậc khác.
>
> **Lập luận đúng nằm ở mức gộp, và nó mạnh hơn nhiều:** ba CI bootstrap trong bảng trên. Bậc 2 và bậc 3 bị chặn chặt quanh 0 trong khi bậc 1 tách hẳn. Đó là bằng chứng về độ lớn hiệu ứng, phát biểu bằng khoảng chứ không bằng phép đếm.

Thứ model thực sự mất khi bị ẩn danh là **tri thức về chiều nhân quả nào hợp lẽ trong thế giới thật**, chứ không phải từ vựng, không phải độ dài ngữ cảnh, không phải khả năng theo dõi ký hiệu lạ.

Hai ô `PSEUDO` - `SYMBOL` đạt p<0,05 đều nằm ở `PROSE` và **ngược chiều nhau** (`nano` +9,43 và `gpt-4.1` -10,47), nên không tạo thành tín hiệu nhất quán.

---

### 7.1 Bậc 1 đổi ba biến cùng lúc, không phải một

Vòng phản biện thứ 6 đo một thứ thang từ vựng ngầm giả định nhưng chưa ai kiểm: **model có nhận ra đề bài đã bị can thiệp không?** Đếm tỉ lệ phản hồi chứa ngôn ngữ tự phát hiện mâu thuẫn (`contradictory`, `inconsistent`, `nonsensical`, `implausible`, `illogical`, `paradox`, `does not make sense`):

| Điều kiện | Model | `KEEP` | **`PERMUTE`** | `SYMBOL` | `PSEUDO` |
|---|---|---|---|---|---|
| RAW | gpt-4.1-nano | 2,3% | **13,2%** | 0,0% | 0,6% |
| RAW | gpt-4.1-mini | 3,4% | **19,0%** | 1,1% | 1,7% |
| RAW | gpt-4.1 | 0,0% | **7,5%** | 0,6% | 1,1% |
| ORACLE | gpt-4.1-nano | 4,0% | **11,5%** | 0,6% | 1,1% |
| ORACLE | gpt-4.1-mini | 4,0% | **19,5%** | 1,7% | 1,1% |
| ORACLE | gpt-4.1 | 2,3% | **8,6%** | 1,7% | 0,0% |

Độ dài phản hồi đi cùng hướng: `mini` ở `RAW` từ 730 lên **1.054** ký tự (+44%), `nano` từ 890 lên **1.311** (+47%), `gpt-4.1` từ 611 lên 727 (+19%). `SYMBOL` và `PSEUDO` thì **ngắn hơn** `KEEP`.

**`PERMUTE` tách hẳn khỏi ba bộ còn lại; `SYMBOL` và `PSEUDO` không phân biệt được với `KEEP`.** Vậy bậc 1 của thang không bỏ đi một thứ mà **ba** thứ:

1. bỏ prior đúng,
2. **gắn một prior sai đối kháng**,
3. **làm item bất thường tới mức model phát hiện được và nói ra**.

Hệ quả cho mục 7: ba con số trong bảng bậc thang vẫn đúng, hai hệ quả loại trừ (độ dài prompt, gánh nặng gắn ký hiệu) vẫn đứng vững. Nhưng **7,53 pp của bậc 1 chưa gán riêng được cho "mất tri thức về chiều nhân quả"** - nó là tổng của ba tác động chưa tách.

Phép kiểm rẻ nhất để tách: một bộ từ vựng thứ năm `IRRELEVANT` (từ thật, miền khác hẳn, không gợi quan hệ nhân quả nào giữa chúng). Khi đó `IRRELEVANT − KEEP` là giá của việc **mất** prior đúng, sạch; `PERMUTE − IRRELEVANT` là giá thêm của việc **bị** prior sai. Ước chừng 3 USD, và nó **có thể thất bại**.

Đây cũng là một phát hiện dùng được ngoài đời: khi văn bản đầu vào mâu thuẫn với tri thức nền, model **tự nói ra**, với tần suất cao gấp 4-11 lần và output dài thêm 19-47%. Cả hai đo được mà không tốn thêm lượt gọi nào - một tín hiệu định tuyến sang người kiểm duyệt.

### 7.2 `SYMBOL` và `PSEUDO` chưa ẩn danh hoàn toàn

`relabel()` trong `src/lexical.py` chỉ thay các cụm nằm trong `variable_mapping` của item. CLadder lại gọi cùng biến đó bằng biến thể ngữ pháp không có trong mapping, nên chúng sống sót.

Đo bằng cách tách từ vựng riêng của story khỏi khuôn mẫu CLadder theo **số story mà một từ xuất hiện** (37 story; từ nằm trong tối đa 3 story là từ vựng riêng):

| Bộ | Item còn residue | Từ sót hay gặp |
|---|---|---|
| `PERMUTE` | 93,1% | *theo thiết kế - derangement dùng lại chính từ vựng của item* |
| **`SYMBOL`** | **61,5%** | `smokers`, `nonsmokers`, `patients`, `students`, `termination`, `letters` |
| **`PSEUDO`** | **61,5%** | như trên |

Cờ `clean` trong `relabel()` không bắt được, vì nó chỉ kiểm cụm-trong-mapping có còn sót không, không kiểm phần tiếng Anh còn lại.

**Hệ quả phải nói rõ:** residue làm `SYMBOL`/`PSEUDO` lệch **về phía** `KEEP`, tức lệch theo đúng hướng **sinh ra** hai kết quả null ở bậc 2 và bậc 3. Cận tương đương ở bảng trên vì thế là cận cho *thao tác ẩn danh một phần đã thực hiện*, không phải cho ẩn danh hoàn toàn. Không làm hỏng kết luận "chi phí trả ở bậc đầu tiên", nhưng phải đi kèm mỗi lần trích con số đó.

Tính bằng `scripts/analyze_anomaly_residue.py`, 0 USD.

---

## 8. Kết quả 6: giá một cạnh sai không đổi theo miền - nhưng phần còn lại chưa xác lập

Thang lỗi đầy đủ `DR`/`ED`/`FE` với k=1,2,3 chạy **hai lần ở n=400 trên cùng bộ item** - một lần với tên biến gốc, một lần thay bằng từ giả. Điều đó tách được hai thứ mà điểm hoà vốn gộp làm một:

- **giá** - pp mất đi trên mỗi cạnh sai, tức độ dốc đường suy giảm
- **ngân sách** - `ORACLE - RAW`, tức chiều cao mà độ dốc đó phải ăn hết

Điểm hoà vốn phụ thuộc cả hai vế, nên k\* đổi có thể do bất kỳ vế nào. Kết quả: **vế giá đứng vững, vế ngân sách thì chưa.**

> **Đính chính công thức, vòng 6.** Văn bản trước viết `k* = ngân sách / giá`. **Đó không phải công thức mã dùng.** `scripts/analyze_types.py:175` giải k\* từ chính đường hồi quy: `(chặn_fit - RAW) / giá`. Hai công thức lệch nhau từ +0,28 tới +2,74 cạnh, vì hệ số chặn fit nằm **dưới** điểm `ORACLE` 1,8-2,2 pp - đường thẳng bị các điểm k=1,2,3 kéo xuống. Nghĩa là "ngân sách hiệu dụng" mà k\* thực sự ăn hết chỉ khoảng 3,0 pp chứ không phải 4,95 pp trên `gpt-4.1`/`KEEP`, **thấp hơn 39%**. Đây mới là lý do chính khiến k\* rơi xuống dưới 1, không phải giá.

### 8.1 Giá KHÔNG đổi - nhưng bảng này CHƯA CÓ SCRIPT SINH RA

> **CẢNH BÁO, vòng 7.** Con số `-0,11 [-1,62 ; 1,42]` và chín CI trên hiệu giá dưới đây chỉ tồn tại ở **một chuỗi in cứng** trong `scripts/compare_price_lexicon.py`. Không có hàm nào bootstrap hiệu độ dốc ghép cặp giữa hai nhánh từ vựng; `price_by_lexicon.csv` chỉ chứa CI của **từng nhánh riêng**.
>
> Đây **đúng khiếm khuyết vòng 6 đã nêu cho mục 8.2 và tuyên bố đã đóng**. Nó vẫn còn, ở mục kề bên, và lần này nó đỡ một kết luận **dương** chứ không phải một kết luận âm.
>
> Phần điểm ước lượng kiểm được: hiệu hai độ dốc trong `types_price_price400*.csv` cho -0,10 (báo cáo ghi -0,11). **Chín cận CI thì không kiểm được bằng bất cứ cách nào.**
>
> **Phải làm trước khi trích mục này ở đâu:** viết `analyze_price_paired.py` bootstrap hiệu độ dốc, bốc cùng bộ item cho hai nhánh. Hoặc rút toàn bộ bảng và câu "cận tương đương 1,6 pp".

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

**Ở n=400: cả ba CI đều chứa 0**, và `gpt-4.1-nano` đi **ngược hướng**.

> **Cập nhật vòng 7: nâng cỡ mẫu, và nó đạt ở 2/3 model.**
>
> | Model | n=400 | n=600 | **Gộp, n≈800 item riêng biệt** |
> |---|---|---|---|
> | gpt-4.1 | +4,71 [-0,79 ; +10,21] p=0,099 | +5,33 [+0,71 ; +9,77] p=0,025 | **+4,67 [+0,86 ; +8,37] p=0,014** |
> | gpt-4.1-mini | +3,50 [-2,16 ; +9,16] p=0,238 | +5,55 [+1,43 ; +9,84] p=0,012 | **+4,38 [+0,88 ; +8,00] p=0,020** |
> | gpt-4.1-nano | -1,75 [-8,45 ; +4,96] | +1,75 [-3,69 ; +7,18] | +1,22 [-3,25 ; +5,69] p=0,615 |
>
> **Đọc kỹ bốn cảnh báo, và đừng gọi đây là nhân bản.**
>
> 1. **Hai mẫu KHÔNG độc lập.** Trùng **34,8%** item, và phần trùng **dùng lại phản hồi cache y hệt từng byte** (tương quan 1,000). Một phần ba mẫu "mới" mang zero thông tin mới.
> 2. **Hai thiết kế khác nhau.** n=400 dùng kmax=3 nên chỉ phủ **7 họ đồ thị**; n=600 dùng kmax=1 và phủ **10 họ**, thêm cả `exp_away` và `collider_bias`. Đó là hai ước lượng của hai đại lượng hơi khác nhau.
> 3. **Cỡ mẫu n≈504 được tính từ độ lớn hiệu ứng quan sát ở một phép kiểm KHÔNG đạt ngưỡng.** Đó là lỗi "power theo hiệu ứng quan sát" kinh điển, cho khoảng 50% công suất nếu ước lượng không chệch, và ước lượng ở n=400 rất nhiễu (SE 2,75 trên điểm 4,71).
> 4. **Hai lần nhìn ở alpha 0,05 không hiệu chỉnh** - sai số họ khoảng 9,75%. Và kết quả n=600 vừa chớm qua ngưỡng (cận dưới +0,71 và +1,43).
>
> Vế bào chữa: mục 8.3 **đã ghi trước** "nâng lên n≈600 cho hai model mạnh" và ghi cả "nó có thể thất bại". Hướng đi được tuyên bố trước, và điều đó có giá trị.
>
> `nano` **đổi dấu** giữa hai mẫu (-1,75 sang +1,75) - bằng chứng trực tiếp rằng đại lượng này chưa ổn định ở cỡ mẫu vài trăm.
>
> **Phát biểu đúng:** *nâng cỡ mẫu từ khoảng 380 lên khoảng 560 item mỗi ô, trên một mẫu mở rộng trùng 24-35% với mẫu cũ. Trên mẫu gộp, hiệu ngân sách đạt ở hai model mạnh. Đây KHÔNG phải một phép nhân bản độc lập, và p chưa hiệu chỉnh cho việc nhìn hai lần.*
>
> **Và một cảnh báo lớn hơn, vòng 7:** ngân sách ở mục này phần lớn là `backadj`. Tách theo nhóm truy vấn trên chính bộ dữ liệu này, điều kiện `KEEP`:
>
> ```
>                     gop    rung-1   backadj   nhan qua
> gpt-4.1            +4,36    0,00    +35,82     -3,68
> gpt-4.1-mini       +7,07   +0,75    +49,25     -3,85
> ```
>
> **Trên nhóm nhân quả thật, ngân sách là ÂM ở 2/3 model.** Nghĩa là "chiều cao mà độ dốc phải ăn hết" phần lớn là việc *xoá đáp án rồi trả lại*, không phải lợi ích suy luận. Mọi k\* đứng trên đại lượng đó. Mục 4.0 dành cả một mục để nói không được gộp như vậy, rồi mục 8 vẫn gộp.

Cỡ mẫu cần để hiệu ngân sách đạt p<0,05: **n≈504** cho `gpt-4.1` (hiện có 382), **n≈948** cho `mini` (hiện có 371), và **n≈5.023** cho `nano` (hiện có 343). Hiệu ứng có thật hay không thì n=400 chưa trả lời được.

> **Sửa ở vòng 6.** Bảng này trước đây là mục duy nhất trong báo cáo **không có script nào sinh ra**, và nó lại chính là mục rút lại một phát hiện tiêu đề. Giờ đã có: `scripts/analyze_budget_paired.py`, tái lập cả sáu giá trị ngân sách chính xác.
>
> Bảng cũng trộn hai cơ sở tính. **Điểm ước lượng** (4,36 / 9,28 ...) tính trên giao `ORACLE` với `RAW` **trong từng nhánh từ vựng riêng**, n = **390 / 388 / 382 / 382 / 371 / 363**. Nhưng **n được trích trong câu trên** lại thuộc cơ sở khác - giao bốn ô qua cả hai nhánh, n = 382 / 371 / 343 - và đó mới là cơ sở duy nhất dùng được cho một phép trừ ghép cặp. Trên cơ sở đó ngân sách là 4,45 / 9,16 (`gpt-4.1`) và 6,47 / 9,97 (`mini`). Cả ba CI vẫn chứa 0, nên kết luận không đổi. Con số cỡ mẫu là **số kế hoạch**, không phải kết quả: nó giả định độ lớn hiệu ứng và phương sai giữ nguyên khi nâng n.

### 8.3 Vậy kết luận được gì

**Được:**

- Giá một cạnh sai **không thay đổi rõ rệt** theo miền từ vựng, với cận tương đương khoảng 1,6 pp mỗi cạnh. Đây là kết quả ghép cặp, CI trên hiệu, không phải suy từ CI chồng nhau.
- `DR` vẫn là loại lỗi đắt nhất ở cả hai chế độ từ vựng, cho cả hai model mạnh.

**Chưa được:**

- Khẳng định "ngân sách tăng gấp đôi khi bỏ neo từ vựng" **chưa xác lập** - cả ba CI chứa 0.
- Do đó khẳng định "điểm hoà vốn dịch từ 0,74 lên 1,93 cạnh" cũng **chưa xác lập**: k\* tính từ `(chặn_fit - RAW) / giá`, và cả chặn lẫn ngân sách đều đứng trên `Delta_struct`, vốn chưa vững.
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

## 9b. Đồ thị vào tới đâu trong chuỗi suy luận

Bốn mục trên đều là bằng chứng **hành vi**: điểm lên khi có đồ thị đúng, xuống khi có đồ thị hỏng. Không cái nào cho biết đồ thị có vào tới tầng **suy luận** hay chỉ tác động tới câu trả lời cuối.

Điều kiện `DR_k1` đảo đúng một cạnh: đồ thị thật nói `u -> v`, prompt khẳng định `v -> u`. Đọc chuỗi và xem nó nói chiều nào. Tính bằng `scripts/analyze_chains.py`, 0 USD.

| Model | theo đồ thị được cấp | giữ chiều thật | không nói |
|---|---|---|---|
| gpt-4.1-nano | 34,5% | 3,4% | 57,5% |
| gpt-4.1-mini | **57,5%** | 1,1% | 33,9% |
| gpt-4.1 | 43,1% | 1,1% | 51,7% |
| **Trung bình** | **45,0%** | **1,9%** | 47,7% |

**Phép kiểm độ đặc hiệu:** dưới `ORACLE` không có cạnh nào bị đảo, và tỉ lệ chuỗi khẳng định chiều ngược là **0,0% ở cả ba model**. Bộ dò không tạo dương tính giả.

### Hai chỗ phải hạ giọng, tìm ra ở vòng 7

**Thứ nhất, con số 1,9% thiếu đường sàn.** Tỉ lệ model **tự nói** chiều `u -> v` khi **không được cấp đồ thị nào** (điều kiện `RAW`):

```
nano 2,9%   mini 4,6%   gpt-4.1 2,3%   (trung binh 3,3%)
```

**1,9% nằm ngay tại sàn đó.** Model gần như không bao giờ tự phát biểu chiều cạnh khi không có khối nào mời nó làm vậy. Nên tương phản "45% so với 1,9%" **không** chứng minh đồ thị ghi đè prior - nó chứng minh **khối văn bản là thứ khiến model phát biểu chiều**.

**Thứ hai, hậu quả tính toán mỏng.**

| Model | acc khi theo đồ thị | acc khi không nói | % đáp án khác với `ORACLE` |
|---|---|---|---|
| nano | 67,2 | 76,8 | 17,5 |
| mini | 76,5 | 84,7 | 22,9 |
| gpt-4.1 | 69,3 | 84,4 | 26,0 |

**75 tới 82% số item mà chuỗi nói rõ chiều SAI vẫn cho đúng đáp án như khi được cấp đồ thị ĐÚNG.** Chiều cạnh vào tới văn bản; nó vào tới phép tính ở khoảng **một phần tư** trường hợp.

Trung vị vị trí của câu khẳng định chiều nằm ở 13% đầu chuỗi, 40% nằm trong 10% đầu - tức nó thường được nhắc ở phần dựng bài rồi ít khi quay lại.

**Điều này giải thích mục 4.3.** Nếu chiều cạnh chỉ đổi đáp án 1/4 số lần thì đương nhiên đồ thị sai gần ngang đồ thị đúng. Hai phát hiện khớp nhau.

**Phát biểu đúng mức:** *chiều cạnh được cấp vào tới văn bản model sinh ra, và đổi đáp án ở khoảng một phần tư trường hợp.* Không phải "bằng chứng cơ chế trực tiếp" theo nghĩa mạnh.

Một giới hạn nữa: gần một nửa số chuỗi không cam kết chiều nào. Con số "không nói" là **cận trên** của sự thờ ơ, không phải phép đo nó - một chuỗi hoàn toàn có thể suy luận trên cạnh đó mà không gọi tên nó ra. Và chiều được đọc bằng biểu thức chính quy, không phải bằng hiểu.

---

## 10. Còn giữ nguyên giá trị

| Hạng mục | Trạng thái |
|---|---|
| Kiểm chứng nhãn chuẩn CLadder bằng solver SCM giải tích | 2.184 model, sai số tối đa 6,66e-16. **Phạm vi: 3/10 họ đồ thị (`chain`, `mediation`, `confounding`) và 1/11 đại lượng (`ATE`).** `ETT`/`NDE`/`NIE` chưa kiểm, mà đó là 31% mẫu thí nghiệm - mở rộng tốn 0 USD |
| Phép gây nhiễu tất định, liệt kê vét cạn | `src/perturb.py`, kiểm bằng NetworkX |
| Seed theo từng `(item, loại, k)` | verify 441/441 ổn định |
| Chấm điểm chỉ trên câu parse được | parse rate báo cáo riêng, **90,8-100%** ở thí nghiệm mới (thấp nhất: `nano`/`PROSE`/`PERMUTE`). Kiểm độ nhạy: chấm câu không parse được thành sai thì tác hại đi từ -10,73 sang -10,79 pp, kết luận không đổi |
| Độ đa dạng của mẫu 174 item | **10 cấu trúc đồ thị, 37 story, 10 query_type, đủ cả ba bậc của Pearl** |
| Cache theo hash `(model, temperature, prompt)` | chạy lại 0 đồng |
| Không có LLM nào tham gia viết nhãn | mọi nhãn lấy nguyên từ CLadder |

---

## 10b. Nhiễm dữ liệu huấn luyện - giả thuyết cạnh tranh, và ba lập luận phản bác

Thêm ở vòng phản biện thứ 6. Sáu vòng trước không nhắc tới chuyện này, dù nó là câu hỏi đầu tiên bất kỳ người phản biện nào cũng sẽ đặt.

**Vấn đề.** CLadder công bố tại NeurIPS 2023 và mở hoàn toàn trên HuggingFace. GPT-4.1 ra sau đó. Nên `full_v1.5_default.csv` gần như chắc chắn nằm trong dữ liệu huấn luyện, kèm nhãn. Nếu vậy thì:

- `KEEP` là điều kiện **duy nhất** trùng khớp nguyên văn với một bộ dữ liệu công khai;
- `PERMUTE`, `SYMBOL`, `PSEUDO` đều phá khớp;
- nên **nhiễm dữ liệu dự đoán chính xác cái hình dạng "toàn bộ chi phí trả ở bậc một"** mà mục 7 quy cho tri thức nhân quả.

Hai giải thích cho việc `gpt-4.1` đạt 75,58% ở `RAW`+`KEEP` - một điều kiện mà chiều nhân quả đã bị xoá khỏi prompt - là (a) model dùng tri thức lẽ thường để suy ra chiều, hoặc (b) model nhớ item. Cả hai dự đoán giống hệt nhau cho mọi bảng ở mục 4 tới 7.

**Ba lập luận phản bác, tất cả đo được từ dữ liệu đã có, 0 USD.**

1. **Không có gom cụm theo story.** ICC theo `story_id` ở `KEEP`/`RAW`: **-0,006 / -0,043 / -0,009** cho ba model, design effect 1,00. Ghi nhớ thường bám theo story - nếu model thuộc bài thì phải thấy ICC dương rõ. Không thấy.

2. **Ô trùng khớp nguyên văn nhất lại hưởng lợi ít nhất từ đồ thị.** Lợi ích của việc có đồ thị trong lời văn (`PROSE` trừ `RAW`), tách theo bộ từ vựng:

   | `KEEP` | `PERMUTE` | `SYMBOL` | `PSEUDO` |
   |---|---|---|---|
   | **+2,09 pp** | +5,33 pp | +7,94 pp | +9,82 pp |

   `PROSE`/`KEEP` là ô duy nhất trùng từng ký tự với CLadder. Nếu nó sống bằng trí nhớ chuỗi, nó phải nổi lên. Nó thấp nhất trong bốn bộ.

3. **`backadj` ở `KEEP`/`RAW` chỉ đạt 41,67%, dưới mức đoán mò** (n=96). Một model nhớ đáp án benchmark không rơi xuống dưới 50% trên cả một loại câu hỏi.

**Kết luận thận trọng.** Ba mảnh trên loại trừ ghi nhớ **theo story** và ghi nhớ **chuỗi nguyên văn**, nhưng **không loại trừ** ghi nhớ phân bố đều toàn bộ dataset. Điều làm hỏng ít nhất là toàn bộ so sánh của dự án đều **ghép cặp trong cùng item**, nên một mức nhiễm đồng đều gần như triệt tiêu khỏi các **hiệu**. Nhưng mọi **mức tuyệt đối** trong báo cáo - và cả điểm hoà vốn k\*, vốn tính từ chiều cao `ORACLE - RAW` - đều đứng trên các mức đó.

**Phép kiểm rẻ nhất còn lại:** kiểm định khả hoán của Oren và cộng sự (ICLR 2024) chạy được trên đúng setup hiện có.

---

## 10c. Nhãn chuẩn của CLadder bám theo một phép tính sai

Đây là phát hiện có giá trị công bố cao nhất của vòng 7, và nó **sửa lại một khẳng định ngược mà báo cáo từng đưa ra**.

### Phát hiện

`scripts/verify_groundtruth.py` dựng lại phân phối đồng thời từ tham số và tính lại mọi đại lượng bằng liệt kê vét cạn: **66.824 phép so sánh trên toàn bộ 7.064 SCM, 0 bỏ sót.**

Chín trên mười họ đồ thị khớp tới 1e-16 trên cả bốn đại lượng nhân quả - **bao gồm ETT, NDE, NIE, vốn chiếm 31% mẫu và chưa từng được ai kiểm.**

Nhưng tám ô không khớp, và chúng có quy luật: CLadder tính bằng cách **nhân xác suất biên của các nút cha như thể chúng độc lập**. Công thức sai khớp **100,0% trên cả 10 họ**; công thức đúng khớp 0%. Ba họ duy nhất mà hai công thức trùng nhau (`chain`, `collision`, `fork`) đúng là ba họ mà các cha của Y vốn độc lập thật.

Họ `arrowhead` sai ở **cả bốn** đại lượng nhân quả (lệch tối đa 0,249), và `arrowhead` chiếm **9/86 = 10,5%** nhóm câu hỏi nhân quả đang đỡ luận điểm tiêu đề.

### Nó có chạm tới nhãn chấm điểm không - CÓ

Lọc riêng các câu mà giá trị đúng và giá trị công bố nằm **hai phía ngưỡng**, tức các câu mà nhãn buộc phải chọn một bên:

| query | n quyết định | nhãn theo giá trị HỎNG | nhãn theo giá trị ĐÚNG |
|---|---|---|---|
| `marginal` | 45 | **45** | 0 |
| `ate` | 9 | **7** | 2 |
| `nde` | 10 | **10** | 0 |
| `nie` | 21 | **20** | 1 |
| **TỔNG** | **85** | **82** | **3** |

**82/85.** Nhãn vàng đi theo giá trị hỏng.

### Thiệt hại cho mẫu của dự án

Gán tỉ lệ lật nhãn theo `query_type` vào thành phần 174 item: **ước 1,4 item, khoảng 0,8%**.

Vì mọi so sánh trong dự án đều ghép cặp trong cùng item nên phần này **gần như triệt tiêu khỏi các hiệu**. Nó **không** triệt tiêu khỏi các **mức tuyệt đối** - và mục 10b đã tự nhận rằng mọi mức tuyệt đối, kể cả k\*, đứng trên đó.

> **Đính chính. Báo cáo từng khẳng định điều ngược lại.**
>
> Bản trước viết rằng nhãn yes/no không bị ảnh hưởng, dẫn một phép kiểm cho 99,05% trên 1.580 câu `marginal`. Phép kiểm đó so nhãn với giá trị in trong chuỗi `reasoning` **đã làm tròn hai chữ số** - mà làm tròn xoá sạch khác biệt đúng ở vùng sát ngưỡng. Mười lăm chỗ lệch được gọi là "nhiễu làm tròn" chính là tín hiệu. Ví dụ thật: exact 0,4978 so với công bố 0,5127 (`mediation`), exact 0,5017 so với 0,4988 (`frontdoor`) - hai phép tính khác nhau cho hai phía ngưỡng, không phải làm tròn.
>
> Tệ hơn: khối `print` mang khẳng định đó là một **chuỗi ký tự cứng**, không phải một phép tính. Đây đúng khiếm khuyết vòng 6 bắt được ở mục 8.2 và tuyên bố đã đóng.

### Vì sao đây là nâng cấp cho danh sách đóng góp

"Ba file `test-*` không chứa câu hỏi" (mục 2) là **lỗi đóng gói**. "Nhãn vàng bám theo một phép tính sai" là **lỗi nội dung**, nặng hơn hẳn, và nó chạm tới mọi công trình đã dùng CLadder.

Kèm theo, một điểm cho khán giả theo khung Pearl: quy ước phân rã hiệu ứng tự nhiên mà CLadder dùng là **both-vs-base** (định nghĩa gốc của Pearl), không phải telescoping. Ở mọi SCM mà câu hỏi quy ước thực sự phát sinh (`mediation`, 1008 ca): both-vs-base 1008/1008, telescoping 0/1008.

---

## 11. Chưa làm

Sáu việc trong bảng cũ **đã làm xong ở vòng 7** và đã chuyển thành kết quả: bộ từ vựng `IRRELEVANT` (mục 7), điều kiện `RAW_INSTR` (mục 4.5), `ED`/`FE` trong thí nghiệm từ vựng, mở rộng `verify_groundtruth.py` lên 10/10 họ và 11/11 đại lượng (mục 10c), phân tích chuỗi suy luận (mục 9b), và nâng n cho phần định giá (mục 8.2). Bảng dưới chỉ còn việc thật sự chưa làm.

| Việc | Vì sao quan trọng | Chi phí |
|---|---|---|
| **Điều kiện `NAMES_ONLY`** | **Việc quan trọng nhất còn lại.** Cùng vị trí, cùng câu lệnh, liệt kê đúng tên biến, **không một mũi tên nào**. Đây là phép kiểm duy nhất có thể cứu lại chữ "đúng" đã rút ở mục 4.3. **Nó có thể thất bại** | 1-2 USD |
| Điều kiện `SCRAMBLE` | Một DAG ngẫu nhiên trên đúng bộ nút, không giữ cạnh nào của đồ thị thật. Bổ sung cho `NAMES_ONLY` | 1-2 USD |
| **Script cho mục 8.1** | Bảng chín CI trên hiệu giá hiện là chuỗi in cứng. Cùng loại: mục 4.1(a), 4.1(b), 4.2, và ICC ở mục 10b | 0 USD |
| **Một họ model thứ hai** | Giới hạn duy nhất đủ nghiêm trọng để chặn công bố. Gemini free tier khảo sát 2026-09-14 và không dùng được: RPD=20 cho hầu hết model. Cần key trả phí | 5-15 USD |
| Một model có suy luận mở rộng | Caliper đã chạy DeepSeek-R1 trên trục P0/P1 (+3,2 pp) nhưng **không có nhánh `ORACLE`**. Câu chưa ai trả lời: model suy luận có dùng được đồ thị cho sẵn không | 20-50 USD |
| Nâng `boot` từ 600 lên 10.000 trong `analyze_types.py` | Hai ô đổi phán quyết theo seed; "6/9 CI chứa 0" thành **4/9** ở 6.000 lần. Con số 6/9 đang được dùng để rút một phát hiện | 20 phút |
| Thống nhất năm hàm `boot()` | Hai ước lượng khác nhau (trung bình-của-trung-bình-ô cho 14,35; trung bình phẳng cho 15,13) và hai kỷ luật RNG khác nhau. Hai script dùng chung một `rng` có trạng thái nên kết quả phụ thuộc thứ tự | 1 giờ |
| Ghi cột `id` của CLadder vào mọi file thô | File thô chỉ có `item` là chỉ số vị trí; `analyze_prior_strength.py` khôi phục bằng cách chạy lại `make_items` và map theo vị trí. Hôm nay đúng, sẽ sai âm thầm nếu bộ lấy mẫu đổi | 30 phút |
| Siết guard `clean` để **bỏ** residue | Hiện đã đo được (50,6% item) và báo cáo, nhưng mặc định là giữ. Bỏ thì đổi mẫu nên phải chạy lại mọi thứ | 0 USD + chạy lại |
| Chấm tay 50 chuỗi ở `ORACLE` so với `DR_k1` | `results/chain_sample_for_hand_coding.md` đã sinh sẵn 50 chuỗi kèm mã tự động. Cần người đọc và xác nhận bộ mã | 0 USD |
| Nhiều lần bốc nhiễu mỗi item (R>=3) | Phương sai do bốc đã đo được 1,36 pp | rẻ |
| Nối lớp structured noise vào pilot | `src/noise.py` đã có 8 loại, chưa nối | rẻ |
| Tính lại đáp án cho `CI_ACTIVE` | Cần dựng SCM mở rộng | vừa |

### Giới hạn cứng

Đồ thị CLadder chỉ 2-5 cạnh. `confounding` và `mediation` không thêm được cạnh nào vì đã là DAG đầy đủ trên 3 nút. GPU trên máy là RTX 3050 Laptop 4GB, không chạy được model 7-8B, nên mọi thứ chạy qua API.

---

## 12. Định vị so với Caliper

Caliper (arXiv:2606.04915, 6/2026) đã công bố rằng ẩn danh tên biến làm tụt độ chính xác trên nhiều model, và báo khoảng cách sụp trên tập pseudoword của CLadder.

> **Cảnh báo về trích dẫn, thêm ở vòng 6.** Toàn bộ phần định vị này đang dựa trên **bản đọc gián tiếp**. Bản PDF Caliper **không có trong kho** (`.gitignore` loại `*.pdf`; hai file PDF duy nhất là NoisyCausal và slides). Và con số không nhất quán: trích dẫn trực tiếp duy nhất trong kho ghi *"the gap collapses by **17x**"* (REVIEW.md mục 3), trong khi ba chỗ khác ghi **"khoảng 19 lần"**. Dải *"7,6 tới 29,6 pp trên 9 model từ 3,8B tới 671B"* **không có nguồn nào trong kho chứng thực**.
>
> **Việc bắt buộc trước khi đưa cho ai đọc:** mở bản PDF, kiểm từng con số, thống nhất 17x hay 19x, và dẫn đúng bảng cho dải phần trăm.

Mức tụt 10-13 pp đo được ở đây **không mâu thuẫn** với những gì Caliper báo. Đó là *nhất quán*, **không phải tái lập** - tái lập đòi hỏi trùng model hoặc trùng giao thức, mà ở đây không có cả hai.

Đóng góp riêng so với Caliper, **viết lại ở vòng 6** sau khi ghế Journal-Fit chỉ ra danh sách cũ vừa thừa vừa thiếu:

1. **Tương tác giữa khối cấu trúc và mức neo từ vựng.** Không phải "cấp đồ thị đúng" - cấp cấu trúc cho model không mới, và chính CLadder đã nhúng DAG vào lời văn (đó là điều kiện `PROSE`). Cái mới là **đo lợi ích của cấu trúc như một hàm của mức neo từ vựng**: +5,98 pp trên 490 item (mục 4.0). **Lưu ý phạm vi, vòng 7:** vế "đúng" đã rút (mục 4.3), và Caliper mục 4.9 đã đo khoảng cách từ vựng dưới can thiệp prompt có cấu trúc - khác biệt thật là **cấu trúc do người cấp** so với **cấu trúc do model tự dựng**.
2. **Đo tỉ lệ đảo chiều cạnh dưới một phép thao tác prior có kiểm soát, ghép cặp, kèm hai đường sàn mô phỏng.** Chấm F1 cạnh trên đồ thị LLM tự trích xuất là cả một dòng nghiên cứu riêng, nên bản thân việc chấm cạnh không phải đóng góp. Phần mới là phép phân ly prior-sai so với prior-vắng-mặt, và con số "đi được 33-55% quãng đường tới tác nhân bỏ hẳn văn bản" (mục 9).
3. **Bậc `PERMUTE` như một đối chứng đối kháng khớp độ dài và khớp bộ chữ.** Cùng bộ từ đến từng chữ cái, dài hơn `KEEP` 9 ký tự, chỉ phá chiều nhân quả. Lập luận loại trừ confound độ dài đi đúng chiều khó.
4. **Lỗi dữ liệu CLadder v1.5** (mục 2): ba file `test-*` không chứa câu hỏi. Kiểm được, hữu ích ngay cho mọi người dùng benchmark này.

> **Đã rút khỏi danh sách đóng góp ở vòng 6:** *"phân rã theo loại lỗi đồ thị, có phân biệt chiều cạnh"*. Theo chính mục 8b, 6/9 CI độ dốc chứa 0, không có giá `FE` nào xác lập ở bất kỳ model nào, và chỉ đúng một điểm hoà vốn có CI không chứa 0. **Không liệt kê một kết quả null làm đóng góp.** Nó là việc đang làm dở, và mục 11 xếp nó đúng chỗ rồi.

Caliper là tiền đề, không phải đối thủ.

---

## 13. Chạy lại

```bash
export PYTHONIOENCODING=utf-8         # bat buoc tren Windows

python scripts/verify_groundtruth.py  # khong can API key
python scripts/feasibility.py         # khong can API key

# ==== PHAN TICH LAI, 0 USD, chay tren du lieu da co trong results/ ====
python scripts/analyze_querygroup.py       # muc 4.0: phan tang + phep kiem tuong tac
python scripts/analyze_budget_paired.py    # muc 8.2: ngan sach ghep cap + co mau can
python scripts/analyze_anomaly_residue.py  # muc 7.1 va 7.2: bat thuong + residue

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

### Thêm ở vòng 7

```bash
# tach cau lenh khoi noi dung do thi (muc 4.5)
python scripts/analyze_instruction.py

# thang tu vung 5 bac, kem IRRELEVANT (muc 7)
python scripts/analyze_ladder5.py

# chuoi suy luan co dung do thi duoc cap khong (muc 9b)
python scripts/analyze_chains.py --dump 50

# gia tung loai loi do thi theo bo tu vung
python scripts/analyze_errortypes_lexical.py

# kiem nhan chuan toan bo 7.064 SCM (muc 10c)
python scripts/verify_groundtruth.py

# ngan sach o n=600
python scripts/analyze_budget_paired.py --prefix n600
```

Toàn bộ 15 script chạy sạch, exit 0, kiểm lần cuối 2026-09-16.
