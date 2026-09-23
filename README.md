# Causal - Neo từ vựng, đồ thị nhân quả, và chi phí của cấu trúc sai

Thực nghiệm về hai câu hỏi:

1. **Mỗi loại lỗi đồ thị nhân quả đắt bao nhiêu** với một LLM đang suy luận, và đồ thị phải sai tới đâu thì việc dùng nó hết có lãi?
2. **Năng lực suy luận nhân quả biểu kiến của LLM có thực chất là năng lực từ vựng hay không**, và nếu đúng thì cấp sẵn đồ thị đúng có bù lại được không?

Xây trên [CLadder](https://github.com/causalNLP/cladder) (Jin et al., NeurIPS 2023). Xuất phát từ [NoisyCausal](https://arxiv.org/abs/2605.04313), định vị cạnh [Caliper](https://arxiv.org/abs/2606.04915).

## Hai kết quả không phụ thuộc cỡ mẫu

Phần thực nghiệm bên dưới chạy trên **một họ model** và n = 174 tới 580 item, nên đọc là giả thuyết có số đỡ. Hai kết quả sau thì khác: chúng đúng hoặc sai, kiểm lại được bằng một lệnh, và không phụ thuộc vào cỡ mẫu, số họ model, hay việc ai đã công bố gì trước.

**1. Một bậc thang liều lượng đo không đúng thứ nó tưởng.** Cách hiển nhiên để hỏi *đồ thị sai nhiều có hại hơn sai ít không* là làm hỏng k = 1, 2, 3 cạnh rồi so. Cách đó hỏng, và hỏng theo kiểu không lộ ra trong bảng nào. Đảo một cạnh có thể **không đổi gì** về mặt nhân quả - nếu tập hiệu chỉnh cửa sau và quan hệ d-separation giữa `X` với `Y` còn nguyên thì đại lượng cần ước lượng không đổi, đáp án đúng cũng không đổi.

| Liều | Nhiễu loạn **không** đổi estimand | Tác hại, chỉ tính item **có** đổi estimand |
|---|---|---|
| `DR_k1` | 45/197 item | **-6,36 pp** |
| `DR_k2` | 0/197 | **-7,36 pp** |
| `DR_k3` | 0/198 | **-6,90 pp** |

Tỷ lệ nhiễu loạn vô hại tụt **23% xuống 0** (45/197 item của tập dùng để tách tác hại), chỉ vì càng hỏng nhiều cạnh thì càng khó hỏng mà không chạm vào đại lượng. Liều lượng vì thế **lẫn hoàn toàn với thành phần mẫu**. Điều kiện hoá trên việc estimand thật sự đổi thì tác hại **phẳng**. Chạy `scripts/classify_perturbations.py`, khớp `results/perturbation_split.csv`.

**2. Năm lỗi trong CLadder v1.5**, kèm lệnh tái lập từng lỗi - xem mục cảnh báo bên dưới và `docs/CLADDER_DATA_ERRORS.md`. Xuất xứ dữ liệu kiểm được từng byte bằng `scripts/verify_data_provenance.py`.

## Kết quả thực nghiệm

**Cấp một khối cấu trúc làm giảm tác hại của việc ẩn danh tên biến, trên nhóm câu hỏi thực sự cần suy luận nhân quả.**

**+5,98 pp, CI 95% [+1,67 ; +10,19], p = 0,0055, n = 490 item** (gộp ba mẫu, bỏ trùng theo id gốc). Khớp `results/structure_arms.csv`.

Ba điều phải đọc kèm. Cả ba tính trên **cùng mẫu gộp n=490** của con số tiêu đề:

| | |
|---|---|
| **Đồ thị có cần ĐÚNG không** | **Phải tách theo nhóm truy vấn, gộp lại là đọc sai.** Với `backadj`, nơi đồ thị LÀ đáp án: **+27,36 đến +43,77 pp**, 6/6 ô. Với suy luận nhân quả thật trên từ vựng quen: đồ thị đúng đáng **khoảng 0** (-5,04 / -0,17 / -0,08, không ô nào đạt ngưỡng), đồ thị đảo một cạnh lấy đi **-8,72 / -6,11 / -5,16**. Thiệt hại xác lập được, lợi ích thì không. Chạy `scripts/analyze_vs_raw.py` |
| **Liều hỏng có đo được không** | **Không như đã tưởng.** 20,6% phép đảo cạnh ở k=1 (82/399 item, toàn mẫu price400) **không đổi ước lượng ATE**, ở k=2 và k=3 thì 0%. Điều kiện trên việc nhiễu thật sự đổi đáp án, tác hại **phẳng**: -6,36 / -7,36 / -6,90. Đường liều cũ là hiệu ứng thành phần mẫu. Chạy `scripts/classify_perturbations.py` |
| **Có "xoá sạch tác hại" không** | **Hiệu ứng gần như toàn bộ là nâng thật.** Đồ thị nâng nhánh ẩn danh +5,12 pp và hạ nhánh `KEEP` chỉ -0,50 pp, tức 9% - mà vế "hạ" không tách khỏi 0 (CI [-3,43 ; +2,36]). Chạy `scripts/analyze_structure_arms.py` |
| **Hiệu ứng nằm ở đâu** | **Không đơn điệu theo độ phức tạp.** Dồn vào nhóm đồ thị 4 cạnh (+9,61 pp, p=0,001, n=230); nhóm 3 cạnh +3,81 và nhóm 2 cạnh +5,30 đều không tách khỏi 0; nhóm 5 cạnh âm (-4,05 pp, n=40). Thứ hạng **giữa từng họ** thì không tái lập được giữa các mẫu. Chạy `scripts/analyze_by_family.py` |

Con số tiêu đề tái lập bằng `scripts/pool_samples.py`, script này cũng phục hồi ánh xạ item sang id gốc CLadder mà `pilot.py` không ghi vào CSV.

**Ba trong bốn giả thuyết cạnh tranh đã loại trừ được:** độ dài prompt (prompt dài hơn làm *tệ* hơn, 8 ô âm có ý nghĩa, 0 dương), một lát cắt may mắn (0/2.000 lần bốc ngẫu nhiên chạm tới mức ban đầu), và độ nhạy chấm điểm. **Giả thuyết thứ tư, residue từ thật còn sót, không kiểm được:** residue trùng khít với loại truy vấn (`backadj` và `correlation` sạch 100%, các loại nhân quả 0-12,5%), nên tách theo residue chính là tách theo loại truy vấn. Chạy `scripts/analyze_falsification.py`.

**Cái gì mất đi khi ẩn danh: tri thức, không phải sự quen mắt.** Bậc thang năm bậc:

| Bậc | Đổi đúng một thứ | Chênh | Trạng thái |
|---|---|---|---|
| `KEEP` sang `IRRELEVANT` | mất prior đúng | **+17,67 pp** [+9,24 ; +26,53] | xác lập |
| `IRRELEVANT` sang `PERMUTE` | bị gán prior **sai** | dưới 8,5 pp | chưa phân giải |
| `IRRELEVANT` sang `SYMBOL` | từ thật sang ký hiệu | dưới 6,37 pp | chưa phân giải |
| `SYMBOL` sang `PSEUDO` | ký hiệu sang từ giả | dưới 6,3 pp | chưa phân giải |

Khi prior đã mất thì từ có thật hay không không còn quan trọng.

**Một phát hiện về chính benchmark:** CLadder tính nhãn chuẩn bằng cách nhân xác suất biên của các nút cha **như thể chúng độc lập**, ở 7/10 họ đồ thị. Trên các câu mà điều đó quyết định nhãn, **83/91 nhãn đi theo giá trị hỏng**. Ước 0,8% mẫu của dự án bị ảnh hưởng; phần này gần như triệt tiêu khỏi các hiệu ghép cặp nhưng không khỏi các mức tuyệt đối. Chạy `scripts/verify_labels.py` để tái lập con số này, `scripts/audit_cladder_arithmetic.py` để xem phạm vi chính xác của lỗi, và `scripts/verify_groundtruth.py` để kiểm lại toàn bộ 7.064 SCM. Đã báo lên nhóm tác giả ngày 20/09/2026: [issue #15](https://github.com/causalNLP/cladder/issues/15) và [#16](https://github.com/causalNLP/cladder/issues/16).

**Mọi nhãn dự án chấm điểm đều đã tính lại độc lập.** Ba loại truy vấn từng bỏ trống - `det-counterfactual`, `collider_bias`, `exp_away`, cộng 1.812 câu và 28 trong 86 item của nhóm nhân quả - nay đã kiểm: **1.812/1.812 nhãn tái lập chính xác**. Chạy `scripts/verify_counterfactual.py`.

n = 490 item ghép cặp qua ba mẫu, 3 model của **một họ** (giới hạn duy nhất đủ nghiêm trọng để chặn công bố), **73.365** lượt chấm điểm (đếm từ `results/*_raw*.csv`, đã trừ dữ liệu cách ly).

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

Năm câu, tất cả rút thẳng từ `results/vs_raw.csv`. Phạm vi: ba checkpoint `gpt-4.1`, CLadder v1.5, chưa có họ model thứ hai.

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

# Không cần API key
python scripts/verify_groundtruth.py   # kiểm chứng 7.064 SCM bằng giải tích
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

- **Một họ model thứ hai** - việc quan trọng nhất còn lại, và là thứ duy nhất chặn công bố mà không có cách lách.
- **`DR_k2` và `DR_k3` trên mẫu n600** - đường liều hiện chỉ có trên 399 item và 7 họ; n600 có 580 item đủ 10 họ.
- **Bậc thang từ vựng trên mẫu n=580** - RQ3 hiện vẫn đứng trên n=85 mỗi ô.
- **Điều kiện `NAMES_ONLY`** - **hạ cấp 2026-09-22**, không còn là phép kiểm sống chết. Mục 4.3b đã bác giả thuyết tái gắn ký hiệu bằng dữ liệu sẵn có: `DR_k1` mang **y hệt** bộ tên biến của `ORACLE` mà giữ được **0** lợi ích. `NAMES_ONLY` giờ là việc củng cố, cho một bậc thang sạch hơn.
- **Điều kiện `SCRAMBLE`** - một DAG ngẫu nhiên trên đúng bộ nút, không giữ cạnh nào của đồ thị thật.
- **Cả ba model đều thuộc dòng GPT-4.1.** Giới hạn duy nhất đủ nghiêm trọng để chặn công bố. Cần ít nhất một dòng khác họ, và một model có suy luận mở rộng kèm nhánh `ORACLE`.
- Điểm hoà vốn `k*` vẫn **chưa xác lập**: vế giá vững, vế ngân sách đạt ở 2/3 model trên mẫu gộp, nhưng `k*` cần cả hai và tử số của nó phần lớn là `backadj`.
- `analyze_types.bootstrap_fit` **đã nâng 600 lên 10.000 vòng** (2026-09-22). Lý do cũ ghi ở đây - "hai ô đổi phán quyết theo seed" - **đã kiểm và KHÔNG tái lập được**: chín ô model x nhánh, năm seed, ở cả 600 lẫn 10.000 vòng đều cho cùng một bộ phán quyết. Việc nâng vẫn đáng làm vì sai số Monte Carlo ở 600 vòng cỡ 0,1 pp, ngang với các hiệu bảng này phải phân xử.

## Cấu trúc

```
src/       lexical.py (đổi từ vựng, 5 bộ), perturb.py (làm hỏng DAG),
           prompts.py (RAW / RAW_INSTR / NAMES_ONLY / ORACLE / PERTURB / PROSE), induce.py,
           noise.py, runner.py (có guard lỗi API), stats.py
scripts/   pilot.py, induction.py (hai script gọi API, tốn tiền), 18 script analyze_*,
           5 script kiểm chứng nhãn và dữ liệu (verify_*, audit_*), 3 cổng
           (check_numbers, check_pipeline_order, verify_determinism), và 7 script
           phụ trợ - tổng 35, mọi script trừ hai cái đầu chạy 0 USD
results/   các bảng CSV kết quả và log chạy thật
```
