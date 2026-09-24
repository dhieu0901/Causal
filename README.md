# Causal - Neo từ vựng, đồ thị nhân quả, và chi phí của cấu trúc sai

Thực nghiệm về hai câu hỏi:

1. **Mỗi loại lỗi đồ thị nhân quả đắt bao nhiêu** với một LLM đang suy luận, và đồ thị phải sai tới đâu thì việc dùng nó hết có lãi?
2. **Năng lực suy luận nhân quả biểu kiến của LLM có thực chất là năng lực từ vựng hay không**, và nếu đúng thì cấp sẵn đồ thị đúng có bù lại được không?

Xây trên [CLadder](https://github.com/causalNLP/cladder) (Jin et al., NeurIPS 2023). Xuất phát từ [NoisyCausal](https://arxiv.org/abs/2605.04313), định vị cạnh [Caliper](https://arxiv.org/abs/2606.04915).

## Hai kết quả không phụ thuộc cỡ mẫu

Phần thực nghiệm bên dưới chạy trên ba model GPT-4.1, **lặp lại trên Llama 3.3 70B** và một phần trên DeepSeek-R1, với n = 86 tới 580 item, nên đọc là giả thuyết có số đỡ. Hai kết quả sau thì khác: chúng đúng hoặc sai, kiểm lại được bằng một lệnh, và không phụ thuộc vào cỡ mẫu, số họ model, hay việc ai đã công bố gì trước.

**1. Một bậc thang liều lượng đo không đúng thứ nó tưởng.** Cách hiển nhiên để hỏi *đồ thị sai nhiều có hại hơn sai ít không* là làm hỏng k = 1, 2, 3 cạnh rồi so. Cách đó hỏng, và hỏng theo kiểu không lộ ra trong bảng nào. Đảo một cạnh có thể **không đổi gì** về mặt nhân quả - nếu tập hiệu chỉnh cửa sau và quan hệ d-separation giữa `X` với `Y` còn nguyên thì đại lượng cần ước lượng không đổi, đáp án đúng cũng không đổi.

| Liều | Phép đảo **không** đổi estimand (toàn mẫu) | Tác hại khi estimand đổi, `price400` | Tác hại khi estimand đổi, `n600` (7 họ) |
|---|---|---|---|
| `DR_k1` | 99/399 (`price400`), 93/420 (`n600`) | **-8,38 pp** | **-8,06 pp** |
| `DR_k2` | 0 | **-7,36 pp** | **-10,63 pp** |
| `DR_k3` | 0 | **-6,90 pp** | **-13,73 pp** |

Tỷ lệ phép đảo vô hại tụt từ **24,8%** (`price400`) và **22,1%** (`n600`, cùng 7 họ) ở k=1 **xuống 0** ở k=2 và k=3, chỉ vì càng hỏng nhiều cạnh thì càng khó hỏng mà không chạm vào đại lượng. Liều lượng vì thế **lẫn với thành phần mẫu** - phần này không phụ thuộc model và tái lập trên hai mẫu. Chạy `scripts/classify_perturbations.py`; số ở `results/perturbation_split_qt.csv` và `results/perturbation_split_n600.csv`, cờ theo loại truy vấn.

**Điều kiện hoá trên estimand đổi thì tác hại có phẳng không: không, câu "phẳng" bị rút.** Trên `price400` thì phẳng; trên `n600` thì tăng theo k. `DR_k1` của `n600` được trả lời khoảng 16/09, `DR_k2` và `DR_k3` vào 24/09, nhưng `scripts/check_drift.py` đã gửi lại các prompt của đợt cũ (24/09/2026, 5,54 USD) và **không thấy độ trôi**: độ chính xác mới trừ cũ là -0,38 [-1,33 ; +0,55] trên 5.256 ô, và không điều kiện hay model nào tách khỏi 0 (`results/drift_check.csv`). Vậy độ vênh không do độ trôi; nó có phải do hai mẫu khác nhau thật không thì chưa nói được (xem ngay dưới). Giữ cố định các item mà cả ba lượt bốc đều đổi estimand, rồi gộp hai mẫu theo id CLadder (`results/perturbation_conditional_slope.csv`): tác hại tăng **-1,82 [-3,86 ; +0,17]** điểm mỗi cạnh với `KEEP` và **-1,81 [-3,67 ; +0,04]** với `PSEUDO`, cả hai sát ngưỡng: p = 0,0680-0,0805 và 0,0495-0,0605 qua năm seed bootstrap (`results/seed_stability.csv`), tức chưa xác lập. Riêng price400: +0,46 (`KEEP`); riêng n600: -3,28 (`KEEP`); hai mẫu khác nhau -3,73 [-7,61 ; +0,15], p = 0,0470-0,0735 qua năm seed, tức nằm trên ranh. Kết luận thận trọng: có thể còn một liều nhỏ cỡ 2 điểm mỗi cạnh sau khi điều kiện hoá, chưa xác lập; phần thành phần mẫu thì đứng. Nhiễu thành phần lớn cỡ nào trên chính dữ liệu này: trên cùng tập item, độ dốc có điều kiện lệch độ dốc chưa điều kiện +1,35 (price400) và +0,08 (n600) điểm mỗi cạnh với `KEEP`, +0,42 và -0,19 với `PSEUDO` - có thật nhưng nhỏ, tối đa khoảng 1,4 điểm mỗi cạnh. Và phép phân loại dự báo đúng hành vi model: ghép cặp trong item, `DR_k1` trừ `ORACLE` trên các item giữ nguyên estimand không tách khỏi 0 ở cả bốn ô (hai mẫu x hai bộ từ vựng), còn trên các item đổi estimand thì âm ở cả bốn, ba ô có ý nghĩa (`results/perturbation_validation.csv`).

*Sửa 24/09/2026: bản trước ghi 45/197, -6,36 và -7,79 / -7,36 / -6,90. Các số ấy tính trên một bản phát lại sai phép bốc nhiễu: `classify_perturbations.py` bốc lại trên đồ thị ký hiệu có cạnh đã sắp xếp, còn `pilot.py` bốc trên đồ thị tên biến theo thứ tự câu văn. Cùng seed, cùng chỉ số, nhưng hai danh sách xếp khác nhau, nên gần một nửa số item bị gán phán quyết của một đồ thị khác với đồ thị model đã thấy. Bản sửa phát lại đúng cách `pilot.py` bốc, và chứng minh bằng cache: 5.814/5.814 prompt dựng lại đều đúng là prompt đã gửi.*

**2. Năm lỗi trong CLadder v1.5**, kèm lệnh tái lập từng lỗi - xem mục cảnh báo bên dưới và `docs/CLADDER_DATA_ERRORS.md`. Xuất xứ dữ liệu kiểm được từng byte bằng `scripts/verify_data_provenance.py`.

## Kết quả thực nghiệm

**Cấp một khối cấu trúc làm giảm tác hại của việc ẩn danh tên biến, trên nhóm câu hỏi thực sự cần suy luận nhân quả.**

**+5,98 pp, CI 95% [+1,67 ; +10,19], p = 0,0055, n = 490 item** (gộp ba mẫu, bỏ trùng theo id gốc). Khớp `results/structure_arms.csv`.

`RAW` vẫn rò cấu trúc ở hai chỗ: phương trình của `det-counterfactual` (không bóc được, vì đó là nội dung câu hỏi) và câu "X is unobserved." ở ba họ có biến ẩn. Với nền `RAW_CLEAN` bỏ câu thứ hai và bỏ luôn các item `det-counterfactual`, con số là **+5,37 [+0,09 ; +10,30], n = 376**, p = 0,0345-0,0500 tuỳ seed bootstrap (`results/raw_clean.csv`, chạy 24/09/2026). Con số tiêu đề sống qua cả hai chỗ rò, nhưng sát ngưỡng.

Ba điều phải đọc kèm. Cả ba tính trên **cùng mẫu gộp n=490** của con số tiêu đề:

| | |
|---|---|
| **Đồ thị có cần ĐÚNG không** | **Phải tách theo nhóm truy vấn, gộp lại là đọc sai.** Với `backadj`, nơi đồ thị LÀ đáp án: **+27,36 đến +43,77 pp**, 6/6 ô. Với suy luận nhân quả thật trên từ vựng quen: đồ thị đúng đáng **khoảng 0** (-5,04 / -0,17 / -0,08, không ô nào đạt ngưỡng), đồ thị đảo một cạnh lấy đi **-8,72 / -6,11 / -5,16**. Thiệt hại xác lập được, lợi ích thì không. Chạy `scripts/analyze_vs_raw.py` |
| **Liều hỏng có đo được không** | **Không như đã tưởng.** 24,8% phép đảo cạnh ở k=1 (99/399 item, toàn mẫu price400) **không đổi ước lượng ATE**, ở k=2 và k=3 thì 0%. Điều kiện trên việc nhiễu thật sự đổi đáp án: price400 **phẳng** (-8,38 / -7,36 / -6,90), n600 **tăng** (-8,06 / -10,63 / -13,73), không do độ trôi. Gộp hai mẫu, giữ cố định item: -1,82 điểm mỗi cạnh, chưa xác lập. Chạy `scripts/classify_perturbations.py` |
| **Có "xoá sạch tác hại" không** | **Hiệu ứng gần như toàn bộ là nâng thật.** Đồ thị nâng nhánh ẩn danh +5,12 pp và hạ nhánh `KEEP` chỉ -0,50 pp, tức 9% - mà vế "hạ" không tách khỏi 0 (CI [-3,43 ; +2,36]). Chạy `scripts/analyze_structure_arms.py` |
| **Hiệu ứng nằm ở đâu** | **Không đơn điệu theo độ phức tạp.** Dồn vào nhóm đồ thị 4 cạnh (+9,61 pp, p=0,001, n=230); nhóm 3 cạnh +3,81 và nhóm 2 cạnh +5,30 đều không tách khỏi 0; nhóm 5 cạnh âm (-4,05 pp, n=40). Thứ hạng **giữa từng họ** thì không tái lập được giữa các mẫu. Chạy `scripts/analyze_by_family.py` |

Con số tiêu đề tái lập bằng `scripts/pool_samples.py`, script này cũng phục hồi ánh xạ item sang id gốc CLadder mà `pilot.py` không ghi vào CSV.

**Ba trong bốn giả thuyết cạnh tranh đã loại trừ được:** độ dài prompt (prompt dài hơn làm *tệ* hơn, 8 ô âm có ý nghĩa, 0 dương), một lát cắt may mắn (0/2.000 lần bốc ngẫu nhiên chạm tới mức ban đầu), và độ nhạy chấm điểm. **Giả thuyết thứ tư, residue từ thật còn sót, không kiểm được:** residue trùng khít với loại truy vấn (`backadj` và `correlation` sạch 100%, các loại nhân quả 0-12,5%), nên tách theo residue chính là tách theo loại truy vấn. Chạy `scripts/analyze_falsification.py`.

**Cái gì mất đi khi ẩn danh: chủ yếu là tri thức, nhưng từ thật không trung tính.** Bậc thang năm bậc:

| Bậc | Đổi đúng một thứ | Mẫu `lex`, 86 item | **Mẫu `n600`, 290 item** | Hai đầu bậc ở `n600` |
|---|---|---|---|---|
| `KEEP` sang `IRRELEVANT` | mất prior đúng | +17,67 [+9,24 ; +26,53] | **+14,99** [+10,41 ; +19,53] | khác đợt chạy |
| `IRRELEVANT` sang `PERMUTE` | bị gán prior **sai** | dưới 8,5 | **-0,83**, dưới 4,49 | cùng đợt |
| `IRRELEVANT` sang `SYMBOL` | từ thật sang ký hiệu | dưới 6,37 | **-3,76** [-6,78 ; -0,82] | cùng đợt |
| `SYMBOL` sang `PSEUDO` | ký hiệu sang từ giả | dưới 6,3 | +2,83 [+0,00 ; +5,63] | khác đợt chạy |
| `KEEP` sang `SYMBOL` | mất prior đúng, đo trên nền ký hiệu trơn | +17,00 [+9,27 ; +25,00] | **+10,93** [+7,17 ; +14,89] | khác đợt chạy |

Trên `n600` (chạy 24/09/2026, `results/ladder5_steps_n600.csv`), bị gán prior sai **không** tốn thêm gì ngoài việc mất prior đúng, với cận tương đương thu từ 8,5 xuống 4,49 điểm. Nhưng câu cũ "khi prior đã mất thì từ có thật hay không không còn quan trọng" **không đứng**: từ thật mà không liên quan làm model trả lời **kém hơn** ký hiệu trơn 3,76 điểm, và bậc này nằm trọn trong một đợt chạy. Độ dài prompt không giải thích được: từ không liên quan thêm trung bình 43 ký tự, nhưng trên 292 item nhóm nhân quả tương quan giữa phần dài thêm và phần mất thêm là +0,069, sai dấu so với một hiệu ứng độ dài (`results/ladder5_length_n600.csv`). Vì `IRRELEVANT` không trung tính, bậc đầu đo trên nó gộp cả giá của từ không liên quan; đo trên nền ký hiệu trơn, mất prior đúng tốn **10,93** điểm, đo trên từ không liên quan thì 14,99. Cách nào thì đó cũng là phần lớn của cú rơi. Bậc đầu tiên vắt qua hai đợt chạy (`KEEP` chạy khoảng 16/09); `scripts/check_drift.py` đã gửi lại các prompt của đợt cũ (24/09/2026, 5,54 USD) và **không thấy độ trôi**: độ chính xác mới trừ cũ là -0,38 [-1,33 ; +0,55] trên 5.256 ô, và không điều kiện hay model nào tách khỏi 0 (`results/drift_check.csv`), nên nó đứng.

**Một phát hiện về chính benchmark:** CLadder tính nhãn chuẩn bằng cách nhân xác suất biên của các nút cha **như thể chúng độc lập**, ở 7/10 họ đồ thị. Trên các câu mà điều đó quyết định nhãn, **83/91 nhãn đi theo giá trị hỏng**. Ước 0,8% mẫu của dự án bị ảnh hưởng; phần này gần như triệt tiêu khỏi các hiệu ghép cặp nhưng không khỏi các mức tuyệt đối. Chạy `scripts/verify_labels.py` để tái lập con số này, `scripts/audit_cladder_arithmetic.py` để xem phạm vi chính xác của lỗi, và `scripts/verify_groundtruth.py` để kiểm lại toàn bộ 7.064 SCM. Đã báo lên nhóm tác giả ngày 20/09/2026: [issue #15](https://github.com/causalNLP/cladder/issues/15) và [#16](https://github.com/causalNLP/cladder/issues/16).

**Mọi nhãn dự án chấm điểm đều đã tính lại độc lập.** Ba loại truy vấn từng bỏ trống - `det-counterfactual`, `collider_bias`, `exp_away`, cộng 1.812 câu và 28 trong 86 item của nhóm nhân quả - nay đã kiểm: **1.812/1.812 nhãn tái lập chính xác**. Chạy `scripts/verify_counterfactual.py`.

n = 490 item ghép cặp qua ba mẫu, 3 model của **một họ** (giới hạn duy nhất đủ nghiêm trọng để chặn công bố), **98.463** lượt chấm điểm (đếm từ `results/*_raw*.csv`, đã trừ dữ liệu cách ly).

## Thiết kế

Bốn bộ từ vựng trên **cùng một item**, giữ nguyên đồ thị, các con số, câu hỏi và nhãn chuẩn. Vì ghép cặp nên McNemar áp dụng trực tiếp.

| Bộ | Ví dụ | Lệch độ dài |
|---|---|---|
| `KEEP` | *Poverty has a direct effect on water quality and cholera* | 0 |
| `PERMUTE` | *Cholera has a direct effect on poverty and water quality* | +9 ký tự |
| `SYMBOL` | *B has a direct effect on A and D* | -169 ký tự |
| `PSEUDO` | *Glimx has a direct effect on muvq and xyfo* | -127 ký tự |

`PERMUTE` hoán vị chính tên biến của item sang vị trí khác trong đồ thị của chính nó bằng một derangement. Bộ từ vựng giống hệt đến từng chữ cái, nên nó là đối chứng độ dài và là bằng chứng rằng thứ bị mất là **tri thức về chiều nhân quả**, không phải từ vựng hay ngữ cảnh.

## Phân tầng nhóm câu hỏi

CLadder trộn ba loại câu hỏi phản ứng khác hẳn nhau với đồ thị. Gộp chung thì hiệu ứng bị pha loãng.

| Nhóm | Tỉ lệ mẫu | Đồ thị làm gì |
|---|---|---|
| `marginal`, `correlation` | 32,2% | Không gì cả. Số học đã có sẵn trên đề bài |
| `backadj` | 18,4% | Đồ thị **chính là** đáp án |
| `ate`, `ett`, `nde`, `nie`, ... | 49,4% | Công cụ dẫn đường cho suy luận nhiều bước |

Tiêu chí thành văn: một loại truy vấn thuộc nhóm nhận dạng nếu đáp án của nó là hàm của riêng đồ thị, độc lập với mọi tham số số học. Bỏ nhóm một vì CLadder cho sẵn **đúng những con số cần dùng**, nên câu hỏi chỉ còn là số học - một sự thật về cách đóng gói prompt, không phải một định lý; tách nhóm hai cũng là một **lập luận**. *(Sửa 23/09/2026 theo REPORT mục 4.0, vòng 7: bản trước dựa việc này vào định lý Causal Hierarchy, nhưng CHT không nói một DAG vô dụng khi tính đại lượng bậc 1.)* Mọi con số tiêu đề đều báo trên nhóm thứ ba.

## Nếu bạn đang dựng pipeline tăng cường bằng đồ thị

Năm câu, tất cả rút thẳng từ `results/vs_raw.csv`. Phạm vi: ba checkpoint `gpt-4.1`, CLadder v1.5. Llama 3.3 70B lặp lại câu về cạnh đảo: với tên quen, đảo một cạnh làm nó mất -5,47 [-9,09 ; -1,84] (`results/second_family.csv`).

| Tình huống | Làm gì | Số đỡ |
|---|---|---|
| Câu hỏi chỉ cần số học trên số đã cho | **Đừng gắn** | 0,00 pp [-1,41 ; +1,41], 0/6 ô đạt ngưỡng |
| Đáp án **chính là** đồ thị (chọn tập hiệu chỉnh, sàng biến) | **Gắn** | +27,36 đến +43,77 pp, 6/6 ô |
| Suy luận nhân quả nhiều bước, tên biến quen thuộc | **Đừng gắn** | đúng: ~0 (0/3 mẫu đạt ngưỡng); đảo 1 cạnh: -5 đến -9 pp (2/3 mẫu) |
| Như trên nhưng tên biến vô nghĩa (mã nội bộ, cột ẩn danh) | **Gắn** | +6,30 pp [+1,95 ; +10,77] |
| Không chắc chiều một cạnh | **Bỏ cạnh, đừng đoán** | thiếu cạnh 0/6 ô hại; đảo cạnh 4/10 ô hại |

Hệ quả đáng chú ý nhất: ở dòng ba, **lợi ích không xác lập được còn thiệt hại thì có**, nên một pipeline tự trích DAG rồi đưa lại cho model có **kỳ vọng âm ở mọi tỉ lệ lỗi trích xuất lớn hơn 0**. Và theo mục 9 của báo cáo, prior sai làm LLM tự trích đồ thị đảo chiều nhiều gấp 5,1 đến 7,7 lần - tức nó làm tăng đúng loại lỗi có tác hại đã xác lập. *(Sửa 23/09/2026: bản trước gọi đảo chiều là loại lỗi "sinh ra nhiều nhất" và "đắt nhất"; thiếu cạnh phổ biến hơn, và thứ bậc giá giữa các loại lỗi đã rút ở REPORT mục 8b.)*

## Cảnh báo cho ai dùng CLadder v1.5

**Cả sáu file `test-*-v1.5.csv` không chứa câu hỏi** - `commonsense`, `anticommonsense`, `noncommonsense`, `balanced`, `easy`, `hard`. 0,00% prompt của cả sáu có dấu `?`, trong khi `full_v1.5_default.csv` là 100%. Chấm điểm trên chúng cho ra đúng 50% và trông y hệt một phát hiện về nhận thức của LLM.

`make_items` trong repo này từ chối chạy trên dữ liệu thiếu câu hỏi.

## Chạy lại

```bash
pip install numpy pandas scipy networkx statsmodels openai
export PYTHONIOENCODING=utf-8          # bắt buộc trên Windows
echo "OPENAI_API_KEY=sk-..." >> .env   # >> để không ghi đè key khác đã có

# Không cần API key - chạy lại MỌI file kết quả theo thứ tự phụ thuộc, rồi ba cổng
bash scripts/run_analysis.sh           # thứ tự đọc từ ORDER trong check_pipeline_order.py

# Hoặc từng bước:
python scripts/verify_groundtruth.py   # kiểm chứng 7.064 SCM bằng giải tích
python scripts/verify_explanations.py  # lỗi 3 và 4: chuỗi giải thích `marginal`, kèm đối chứng nhãn
python scripts/verify_labels.py        # nhãn yes/no đi theo giá trị nào
python scripts/verify_counterfactual.py   # bậc 3 và va chạm: 1.812 nhãn còn lại
python scripts/audit_cladder_arithmetic.py  # phạm vi chính xác của lỗi CLadder
python scripts/pool_samples.py         # gộp ba mẫu, tái lập con số tiêu đề
python scripts/analyze_structure_arms.py  # các nhánh ORACLE/PROSE/DR_k1, đo bằng DiD
python scripts/analyze_vs_raw.py       # KẾT QUẢ CHÍNH: mọi nhánh so với RAW, tách theo nhóm truy vấn
python scripts/classify_perturbations.py  # bao nhiêu phép nhiễu thật sự đổi đáp án
python scripts/analyze_by_family.py    # DiD theo từng họ đồ thị, theo nút, theo cạnh
python scripts/analyze_dose.py         # liều hay cấu trúc mới là biến giải thích
python scripts/make_figures.py         # sinh hình cho slide từ CSV
python scripts/feasibility.py          # khả thi, độ phân giải mẫu, dự toán

# Hai cổng chặn - chạy trước khi tin bất kỳ con số nào
python scripts/check_numbers.py        # mọi đại lượng pp, và mọi CI cùng ước lượng của nó, có truy được về một dòng CSV không
python scripts/verify_determinism.py   # chạy lại có ra đúng file cũ không (--all cho đầy đủ)

# Thí nghiệm từ vựng ghép cặp - kết quả chính
for LEX in KEEP PERMUTE SYMBOL PSEUDO; do
  python scripts/pilot.py --n 200 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
      --kmax 1 --types DR --drop-nonsense --lexicon $LEX --tag "_lex$LEX"
done
python scripts/analyze_lexical.py

# Định giá từng loại lỗi đồ thị
python scripts/pilot.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1 \
                        --kmax 3 --types DR,ED,FE
python scripts/induction.py --n 150 --models gpt-4.1-nano,gpt-4.1-mini,gpt-4.1
python scripts/analyze_types.py
```

Mọi lượt gọi được cache theo `sha256(model, temperature, prompt)`, nên chạy lại không tốn thêm tiền và một lần chạy bị ngắt sẽ tiếp tục từ chỗ dừng.

Dữ liệu CLadder không nằm trong repo, và phải tải từ **cả hai** nguồn chứ không phải một trong hai - hai kênh mang hai bộ file khác nhau:

| File trong `data/` | Nguồn | Tên gốc |
|---|---|---|
| `cladder-meta.json` | [causalNLP/cladder](https://github.com/causalNLP/cladder) `data/` | `cladder-v1-meta-models.json` |
| `cladder-questions.json` | cùng trên | `cladder-v1-questions.json` |
| `full_v1.5_default.csv` | [HuggingFace](https://huggingface.co/datasets/causal-nlp/CLadder) `data/` | giữ nguyên tên |
| `test-*-v1.5.csv` (6 file) | cùng trên | giữ nguyên tên |

Hai file JSON **chỉ có trên GitHub**, các file CSV **chỉ có trên HuggingFace**. Dự án đổi tên hai file JSON khi tải về.

Chạy `python scripts/verify_data_provenance.py` để đối chiếu từng byte với bản phát hành gốc - script so git blob SHA-1 của cả 9 file, và đo lại tỷ lệ prompt có dấu hỏi thay vì trích lại con số. Thoát mã 1 nếu lệch.

## Việc chưa xong

- ~~Một họ model thứ hai~~ - **đã chạy 24/09/2026 qua OpenRouter, 4,19 USD** (`scripts/run_llama70b.sh`, `scripts/run_r1.sh`, `scripts/analyze_second_family.py`). Llama 3.3 70B trên đúng các prompt GPT-4.1 đã nhận: DiD gộp **+7,24 [+1,49 ; +13,09]**, n=457, so với +5,98 của GPT-4.1; hiệu giữa hai họ, ghép cặp, +0,80 [-5,41 ; +6,95]. Bậc thang từ vựng tái lập, trừ bậc từ giả: Llama mất +7,72 [+3,51 ; +11,94] so với ký hiệu, GPT-4.1 chỉ +2,83. DeepSeek-R1 trên 86 item ẩn danh: đồ thị đúng hơn đồ thị đảo một cạnh **+23,26 [+12,79 ; +33,72]**, và hơn không đồ thị +15,12 [+4,65 ; +25,58] - hiệu sau dương ở mọi cách xử lý nhưng mất ý nghĩa khi chỉ giữ 37 item có đáp án về đúng chỗ. Chi tiết ở REPORT mục 4b.
- ~~Kiểm độ trôi của model được phục vụ~~ - **đã chạy 24/09/2026, 5,54 USD, không thấy độ trôi** (`scripts/check_drift.py`, `results/drift_check.csv`). Phát hiện phụ của cùng phép kiểm: ở nhiệt độ 0, **15,0%** câu trả lời bị lật khi hỏi lại (nano 22,4%, mini 12,6%, gpt-4.1 10,1%) dù độ chính xác không đổi.
- ~~`DR_k2` và `DR_k3` trên mẫu n600~~ - **đã chạy 24/09/2026**, xem Kết quả 1 ở trên.
- ~~Bậc thang từ vựng trên mẫu n=580~~ - **đã chạy 24/09/2026** ở điều kiện `RAW`, xem bảng bậc thang.
- ~~Điều kiện `NAMES_ONLY`~~ và ~~`SCRAMBLE`~~ - **đã chạy 24/09/2026 trên n600**, `analyze_structure_arms.py` mục 6. Cùng đợt chạy: cạnh ngẫu nhiên hại hơn không có cạnh nào, `SCRAMBLE` trừ `NAMES_ONLY` = -13,95 (`KEEP`) và -7,96 (`PSEUDO`); `SCRAMBLE` ngang `DR_k3`. Khác đợt: trên `PSEUDO`, `ORACLE` trừ `NAMES_ONLY` = +4,34 [+0,12 ; +8,58].
- Toàn bộ đợt 24/09 tốn **15,62 USD** thật (ước tính trước khi chạy: 16,11), 0 lỗi API, lệnh ở `scripts/run_n600_extensions.sh`; kiểm độ trôi thêm **5,54 USD** (ước tính 5,58).
- **R1 mới chạy trên `PSEUDO`.** Chưa có DiD cho model suy luận; chạy thêm `KEEP` tốn khoảng 1,8 USD. R1 ở nhiệt độ 0,6 nên mỗi câu trả lời là một lần rút.
- Điểm hoà vốn `k*` vẫn **chưa xác lập**: vế giá vững, vế ngân sách đạt ở 2/3 model trên mẫu gộp, nhưng `k*` cần cả hai và tử số của nó phần lớn là `backadj`.
- `analyze_types.bootstrap_fit` **đã nâng 600 lên 10.000 vòng** (2026-09-22). Lý do cũ ghi ở đây - "hai ô đổi phán quyết theo seed" - **đã kiểm và KHÔNG tái lập được**: chín ô model x nhánh, năm seed, ở cả 600 lẫn 10.000 vòng đều cho cùng một bộ phán quyết. Việc nâng vẫn đáng làm vì sai số Monte Carlo ở 600 vòng cỡ 0,1 pp, ngang với các hiệu bảng này phải phân xử.

## Cấu trúc

```
src/       lexical.py (đổi từ vựng, 5 bộ), perturb.py (làm hỏng DAG),
           prompts.py (RAW / RAW_INSTR / NAMES_ONLY / ORACLE / PERTURB / PROSE), induce.py,
           noise.py, runner.py (có guard lỗi API), stats.py
scripts/   pilot.py, induction.py, check_drift.py (ba script gọi API, tốn tiền;
           cả ba có --dry-run hoặc cache), 18 script analyze_*,
           6 script kiểm chứng nhãn và dữ liệu (verify_*, audit_*), 4 cổng
           (check_consistency, check_numbers, check_pipeline_order,
           verify_determinism), audit_cache_agreement.py (so mọi dòng CSV với
           cache; cần cache nên chạy tay), backfill_ids.py (ghi và kiểm cột id
           CLadder trong file thô), ci_active_gold.py (đáp án cho CI_ACTIVE), và
           7 script phụ trợ - tổng 41, mọi script trừ ba cái đầu chạy 0 USD.
           run_analysis.sh chạy lại toàn bộ phần phân tích, không gọi API;
           run_n600_extensions.sh và run_clean_raw.sh là hai đợt tốn credit
           ngày 24/09/2026
results/   các bảng CSV kết quả và log chạy thật
```
