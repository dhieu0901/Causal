# Causal - Neo từ vựng, đồ thị nhân quả, và chi phí của cấu trúc sai

Thực nghiệm về hai câu hỏi:

1. **Mỗi loại lỗi đồ thị nhân quả đắt bao nhiêu** với một LLM đang suy luận, và đồ thị phải sai tới đâu thì việc dùng nó hết có lãi?
2. **Năng lực suy luận nhân quả biểu kiến của LLM có thực chất là năng lực từ vựng hay không**, và nếu đúng thì cấp sẵn đồ thị đúng có bù lại được không?

Xây trên [CLadder](https://github.com/causalNLP/cladder) (Jin et al., NeurIPS 2023). Xuất phát từ [NoisyCausal](https://arxiv.org/abs/2605.04313), định vị cạnh [Caliper](https://arxiv.org/abs/2606.04915).

## Kết quả chính

**Cấp một khối cấu trúc làm giảm tác hại của việc ẩn danh tên biến, trên nhóm câu hỏi thực sự cần suy luận nhân quả.**

**+5,98 pp, CI 95% [+1,78 ; +10,30], p = 0,005, n = 490 item** (gộp ba mẫu, bỏ trùng theo id gốc).

Ba điều phải đọc kèm:

| | |
|---|---|
| **Đồ thị có cần ĐÚNG không** | **Chưa chứng minh được.** Đồ thị đảo chiều một cạnh cũng đạt ngưỡng: +8,64 pp [+1,81 ; +16,38], p=0,018. Hiệu giữa hai bên không tách khỏi 0 |
| **Có "xoá sạch tác hại" không** | **Không.** Đồ thị nâng nhánh ẩn danh +9,75 pp nhưng hạ nhánh `KEEP` -4,60 pp. **32% hiệu ứng là do làm hại điều kiện vốn đang ổn** |
| **Hiệu ứng nằm ở đâu** | Tập trung ở đồ thị từ 4 nút trở lên (+22,44 pp, p=0,0005). Trên đồ thị 3 nút thì không tách khỏi 0 (+5,72 pp, p=0,217) |

**Bốn giả thuyết cạnh tranh đã loại trừ được:** độ dài prompt (prompt dài hơn làm *tệ* hơn, 8 ô âm có ý nghĩa, 0 dương), residue từ thật còn sót (làm *co* hiệu ứng), một lát cắt may mắn (0/2.000 lần bốc ngẫu nhiên chạm tới mức ban đầu), và độ nhạy chấm điểm.

**Cái gì mất đi khi ẩn danh: tri thức, không phải sự quen mắt.** Bậc thang năm bậc:

| Bậc | Đổi đúng một thứ | Chênh | Trạng thái |
|---|---|---|---|
| `KEEP` sang `IRRELEVANT` | mất prior đúng | **+17,67 pp** [+9,24 ; +26,53] | xác lập |
| `IRRELEVANT` sang `PERMUTE` | bị gán prior **sai** | dưới 8,5 pp | chưa phân giải |
| `IRRELEVANT` sang `SYMBOL` | từ thật sang ký hiệu | dưới 6,4 pp | chưa phân giải |
| `SYMBOL` sang `PSEUDO` | ký hiệu sang từ giả | dưới 6,3 pp | chưa phân giải |

Khi prior đã mất thì từ có thật hay không không còn quan trọng.

**Một phát hiện về chính benchmark:** CLadder tính nhãn chuẩn bằng cách nhân xác suất biên của các nút cha **như thể chúng độc lập**, ở 7/10 họ đồ thị. Trên các câu mà điều đó quyết định nhãn, **82/85 nhãn đi theo giá trị hỏng**. Ước 0,8% mẫu của dự án bị ảnh hưởng; phần này gần như triệt tiêu khỏi các hiệu ghép cặp nhưng không khỏi các mức tuyệt đối. Chạy `scripts/verify_groundtruth.py` để tái lập.

n = 490 item ghép cặp qua ba mẫu, 3 model của **một họ** (giới hạn duy nhất đủ nghiêm trọng để chặn công bố), hơn 60.000 lượt gọi API.

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
python scripts/verify_groundtruth.py   # kiểm chứng nhãn CLadder bằng giải tích
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
