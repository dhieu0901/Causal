# Causal - Neo từ vựng, đồ thị nhân quả, và chi phí của cấu trúc sai

Thực nghiệm về hai câu hỏi:

1. **Mỗi loại lỗi đồ thị nhân quả đắt bao nhiêu** với một LLM đang suy luận, và đồ thị phải sai tới đâu thì việc dùng nó hết có lãi?
2. **Năng lực suy luận nhân quả biểu kiến của LLM có thực chất là năng lực từ vựng hay không**, và nếu đúng thì cấp sẵn đồ thị đúng có bù lại được không?

Xây trên [CLadder](https://github.com/causalNLP/cladder) (Jin et al., NeurIPS 2023). Xuất phát từ [NoisyCausal](https://arxiv.org/abs/2605.04313), định vị cạnh [Caliper](https://arxiv.org/abs/2606.04915).

## Kết quả chính

**Cấp đồ thị nhân quả đúng cắt gần một nửa tác hại của việc ẩn danh tên biến.**

| Điều kiện | Số ô đạt p<0,05 | Tác hại trung bình |
|---|---|---|
| Không đồ thị (`RAW`) | **8/9** | **-10,73 pp** |
| Có đồ thị đúng (`ORACLE`) | **3/9** | **-5,67 pp** |

**Toàn bộ chi phí đó trả ở bậc đầu tiên của bậc thang từ vựng**, không phải ở độ dài prompt hay việc gắn ký hiệu:

| Bậc | Bỏ đi thêm cái gì | Chênh TB | p<0,05 |
|---|---|---|---|
| `PERMUTE` - `KEEP` | chiều nhân quả hợp lẽ thường | **-7,52 pp** | **5/12** |
| `SYMBOL` - `PERMUTE` | mất hẳn từ thật (prompt ngắn 169 ký tự) | +0,73 pp | 0/12 |
| `PSEUDO` - `SYMBOL` | ký hiệu khó phân biệt | +0,13 pp | 2/12 |

n=174 item ghép cặp, 3 model, 19.901 lượt gọi API.

## Thiết kế

Bốn bộ từ vựng trên **cùng một item**, giữ nguyên đồ thị, các con số, câu hỏi và nhãn chuẩn. Vì ghép cặp nên McNemar áp dụng trực tiếp.

| Bộ | Ví dụ | Lệch độ dài |
|---|---|---|
| `KEEP` | *Poverty has a direct effect on water quality and cholera* | 0 |
| `PERMUTE` | *Cholera has a direct effect on poverty and water quality* | +9 ký tự |
| `SYMBOL` | *B has a direct effect on A and D* | -169 ký tự |
| `PSEUDO` | *Glimx has a direct effect on muvq and xyfo* | -127 ký tự |

`PERMUTE` hoán vị chính tên biến của item sang vị trí khác trong đồ thị của chính nó bằng một derangement. Bộ từ vựng giống hệt đến từng chữ cái, nên nó là đối chứng độ dài và là bằng chứng rằng thứ bị mất là **tri thức về chiều nhân quả**, không phải từ vựng hay ngữ cảnh.

## Cảnh báo cho ai dùng CLadder v1.5

**Ba file `test-commonsense`, `test-anticommonsense`, `test-noncommonsense` không chứa câu hỏi.** 0,00% prompt của chúng có dấu `?`, trong khi `full_v1.5_default.csv` là 100%. Chấm điểm trên chúng cho ra đúng 50% và trông y hệt một phát hiện về nhận thức của LLM.

Dự án này đã mắc đúng lỗi đó và báo cáo nó suốt nhiều ngày trước khi phát hiện. `make_items` giờ từ chối chạy trên dữ liệu thiếu câu hỏi. Chi tiết ở [REPORT.md](REPORT.md) mục 2.

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

## Tài liệu

| File | Nội dung |
|---|---|
| [REPORT.md](REPORT.md) | Báo cáo kết quả, kèm phần đính chính những gì đã bị rút và vì sao |
| [WALKTHROUGH.md](WALKTHROUGH.md) | Cẩm nang toàn dự án, gồm sổ tay 8 cái bẫy kỹ thuật |
| [REVIEW.md](REVIEW.md) | Phản biện hội đồng 5 ghế, hai vòng, có cả phần tự rút lại của chính bản phản biện |

## Việc chưa xong

- Định giá từng loại lỗi đồ thị **chưa xác lập được**: 6/9 ô có CI độ dốc còn chứa 0. Cần cỡ mẫu lớn hơn.
- Nửa "neo từ vựng giúp trích xuất đồ thị" vẫn là so sánh between-items. Cần chạy `induction.py` trên cả bốn bộ ghép cặp.
- Cả ba model đều thuộc dòng GPT-4.1. Cần ít nhất một dòng khác họ.

## Cấu trúc

```
src/       lexical.py (đổi từ vựng trong item), perturb.py (làm hỏng DAG),
           prompts.py, induce.py, noise.py, runner.py, stats.py
scripts/   pilot.py, induction.py, analyze_lexical.py, analyze_types.py,
           verify_groundtruth.py, feasibility.py
results/   các bảng CSV kết quả và log chạy thật
```

Các file mang tiền tố `INVALID_no_question_` là kết quả hỏng, giữ lại để đối chiếu chứ không dùng để báo cáo.
