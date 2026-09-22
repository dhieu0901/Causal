# Causal - Neo từ vựng, đồ thị nhân quả, và chi phí của cấu trúc sai

Thực nghiệm về hai câu hỏi:

1. **Mỗi loại lỗi đồ thị nhân quả đắt bao nhiêu** với một LLM đang suy luận, và đồ thị phải sai tới đâu thì việc dùng nó hết có lãi?
2. **Năng lực suy luận nhân quả biểu kiến của LLM có thực chất là năng lực từ vựng hay không**, và nếu đúng thì cấp sẵn đồ thị đúng có bù lại được không?

Xây trên [CLadder](https://github.com/causalNLP/cladder) (Jin et al., NeurIPS 2023). Xuất phát từ [NoisyCausal](https://arxiv.org/abs/2605.04313), định vị cạnh [Caliper](https://arxiv.org/abs/2606.04915).

## Kết quả chính

**Cấp một khối cấu trúc làm giảm tác hại của việc ẩn danh tên biến, trên nhóm câu hỏi thực sự cần suy luận nhân quả.**

**+5,98 pp, CI 95% [+1,78 ; +10,30], p = 0,005, n = 490 item** (gộp ba mẫu, bỏ trùng theo id gốc).

Ba điều phải đọc kèm. Cả ba tính trên **cùng mẫu gộp n=490** của con số tiêu đề:

| | |
|---|---|
| **Đồ thị có cần ĐÚNG không** | **Phải tách theo nhóm truy vấn, gộp lại là đọc sai.** Với `backadj`, nơi đồ thị LÀ đáp án: **+27,36 đến +43,77 pp**, 6/6 ô. Với suy luận nhân quả thật trên từ vựng quen: đồ thị đúng đáng **khoảng 0** (-5,04 / -0,17 / -0,08, không ô nào đạt ngưỡng), đồ thị đảo một cạnh lấy đi **-8,72 / -6,11 / -5,16**. Thiệt hại xác lập được, lợi ích thì không. Chạy `scripts/analyze_vs_raw.py` |
| **Liều hỏng có đo được không** | **Không như đã tưởng.** 20,6% phép đảo cạnh ở k=1 **không đổi ước lượng ATE**, ở k=2 và k=3 thì 0%. Điều kiện trên việc nhiễu thật sự đổi đáp án, tác hại **phẳng**: -6,36 / -7,36 / -6,90. Đường liều cũ là hiệu ứng thành phần mẫu. Chạy `scripts/classify_perturbations.py` |
| **Có "xoá sạch tác hại" không** | **Hiệu ứng gần như toàn bộ là nâng thật.** Đồ thị nâng nhánh ẩn danh +5,12 pp và hạ nhánh `KEEP` chỉ -0,50 pp, tức 9% - mà vế "hạ" không tách khỏi 0 (CI [-3,43 ; +2,36]). Chạy `scripts/analyze_structure_arms.py` |
| **Hiệu ứng nằm ở đâu** | **Không đơn điệu theo độ phức tạp.** Dồn vào nhóm đồ thị 4 cạnh (+9,61 pp, p=0,001, n=230); nhóm 3 cạnh +3,81 và nhóm 2 cạnh +5,30 đều không tách khỏi 0; nhóm 5 cạnh âm (-4,05 pp, n=40). Thứ hạng **giữa từng họ** thì không tái lập được giữa các mẫu. Chạy `scripts/analyze_by_family.py` |

Con số tiêu đề tái lập bằng `scripts/pool_samples.py`, script này cũng phục hồi ánh xạ item sang id gốc CLadder mà `pilot.py` không ghi vào CSV.

**Bốn giả thuyết cạnh tranh đã loại trừ được:** độ dài prompt (prompt dài hơn làm *tệ* hơn, 8 ô âm có ý nghĩa, 0 dương), residue từ thật còn sót (làm *co* hiệu ứng), một lát cắt may mắn (0/2.000 lần bốc ngẫu nhiên chạm tới mức ban đầu), và độ nhạy chấm điểm.

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

Tiêu chí thành văn: một loại truy vấn thuộc nhóm nhận dạng nếu đáp án của nó là hàm của riêng đồ thị, độc lập với mọi tham số số học. Bỏ nhóm một là do **định lý** Causal Hierarchy; tách nhóm hai là một **lập luận**, không phải định lý. Mọi con số tiêu đề đều báo trên nhóm thứ ba.

## Cảnh báo cho ai dùng CLadder v1.5

**Ba file `test-commonsense`, `test-anticommonsense`, `test-noncommonsense` không chứa câu hỏi.** 0,00% prompt của chúng có dấu `?`, trong khi `full_v1.5_default.csv` là 100%. Chấm điểm trên chúng cho ra đúng 50% và trông y hệt một phát hiện về nhận thức của LLM.

`make_items` trong repo này từ chối chạy trên dữ liệu thiếu câu hỏi.

## Chạy lại

```bash
pip install numpy pandas scipy networkx statsmodels openai
export PYTHONIOENCODING=utf-8          # bắt buộc trên Windows
echo "OPENAI_API_KEY=sk-..." > .env

# Không cần API key
python scripts/verify_groundtruth.py   # kiểm chứng 7.064 SCM bằng giải tích
python scripts/verify_labels.py        # nhãn yes/no đi theo giá trị nào
python scripts/verify_counterfactual.py   # bậc 3 và va chạm: 1.812 nhãn còn lại
python scripts/audit_cladder_arithmetic.py  # phạm vi chính xác của lỗi CLadder
python scripts/pool_samples.py         # gộp ba mẫu, tái lập con số tiêu đề
python scripts/analyze_structure_arms.py  # các nhánh ORACLE/PROSE/DR_k1
python scripts/analyze_by_family.py    # DiD theo từng họ đồ thị, theo nút, theo cạnh
python scripts/analyze_dose.py         # liều hay cấu trúc mới là biến giải thích
python scripts/make_figures.py         # sinh hình cho slide từ CSV
python scripts/feasibility.py          # khả thi, độ phân giải mẫu, dự toán

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

Dữ liệu CLadder không nằm trong repo. Tải từ [causalNLP/cladder](https://github.com/causalNLP/cladder) hoặc [HuggingFace](https://huggingface.co/datasets/causal-nlp/CLadder) vào `data/`.

## Việc chưa xong

- **Điều kiện `NAMES_ONLY`** - việc quan trọng nhất còn lại. Khối liệt kê đúng tên biến nhưng **không một mũi tên nào**. Đây là phép kiểm duy nhất có thể cứu lại chữ "đúng" trong "đồ thị đúng", **và nó có thể thất bại**.
- **Điều kiện `SCRAMBLE`** - một DAG ngẫu nhiên trên đúng bộ nút, không giữ cạnh nào của đồ thị thật.
- **Bảng chín CI trên hiệu giá chưa có script sinh ra** - hiện là chuỗi in cứng trong `scripts/compare_price_lexicon.py`.
- **Cả ba model đều thuộc dòng GPT-4.1.** Giới hạn duy nhất đủ nghiêm trọng để chặn công bố. Cần ít nhất một dòng khác họ, và một model có suy luận mở rộng kèm nhánh `ORACLE`.
- Điểm hoà vốn `k*` vẫn **chưa xác lập**: vế giá vững, vế ngân sách đạt ở 2/3 model trên mẫu gộp, nhưng `k*` cần cả hai và tử số của nó phần lớn là `backadj`.
- `analyze_types.bootstrap_fit` chạy 600 lần - quá thô, hai ô đổi phán quyết theo seed.

## Cấu trúc

```
src/       lexical.py (đổi từ vựng, 5 bộ), perturb.py (làm hỏng DAG),
           prompts.py (RAW / RAW_INSTR / ORACLE / PERTURB), induce.py,
           noise.py, runner.py (có guard lỗi API), stats.py
scripts/   pilot.py, induction.py, và 15 script phân tích
results/   các bảng CSV kết quả và log chạy thật
```
