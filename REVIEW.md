# BÁO CÁO PHẢN BIỆN - Hội đồng 5 ghế

**Đối tượng thẩm định:** dự án `Noisy Causal` (REPORT.md + WALKTHROUGH.md + toàn bộ mã nguồn và các bảng kết quả)
**Chế độ:** `academic-paper-reviewer` v1.11.1, mode `full`, sau đó `re-review`
**Ngày:** 2026-09-07, cập nhật 2026-09-08
**Quyết định vòng 1:** **MAJOR REVISION**
**Quyết định vòng 2 (sau khi tác giả sửa):** **MINOR REVISION** - xem mục 9
**Cập nhật sau vòng 2:** R2 và R4 đã hoàn thành bằng bộ từ vựng `PERMUTE` - xem mục 10
**Quyết định vòng 3 (sau khi chạy induction ghép cặp):** **MINOR REVISION** - xem mục 11
**Vòng 4 (kiểm dữ liệu theo yêu cầu, đối chiếu bài CLadder gốc):** **MINOR REVISION** - một finding MAJOR mới, xem mục 12

---

## ĐÍNH CHÍNH CỦA CHÍNH BẢN PHẢN BIỆN NÀY

Vòng 1 nêu một finding CRITICAL (`C1`) như sau: *mức rơi trên `correlation` (33 tới 54 pp) lớn ngang hoặc hơn trên `ate`, mà `correlation` là câu thuần số học, nên thứ bị phá huỷ không phải năng lực nhân quả.*

**Con số đó sai, và cách diễn giải cũng sai.** Hai lý do:

1. Tôi ánh xạ `item` của `pilot_raw.csv` bằng `make_items(147, seed=0, kmax=3, "test-commonsense-v1.5.csv")`. Cấu hình thật của lần chạy là `seed=20260907, kmax=3, "full_v1.5_default.csv"`. Nhánh pseudoword thật là `kmax=1` cho 145 item, không phải 147. Toàn bộ bảng phân rã theo `query_type` của vòng 1 gán nhầm item.
2. Nguyên nhân thật nằm sâu hơn nhiều và không ghế nào trong vòng 1 tìm ra: **các file `test-*-v1.5.csv` không chứa câu hỏi**. Mức 50% là hệ quả của việc buộc đoán, không phải của việc mất năng lực nhân quả cũng không phải của việc khó gắn ký hiệu.

Finding `C1` được rút. Nó được thay bằng `C1'` ở mục 9, có mức độ nghiêm trọng cao hơn.

Hai kiểm chứng khác của vòng 1 - gán lại trọng số `query_type` chỉ dịch 0,98 pp, và ICC theo story xấp xỉ 0 - cũng dùng ánh xạ sai nên **không còn giá trị**. Điều đó không thay đổi kết luận nào, vì cả hai nhánh chúng bênh vực nay đều đã bị rút vì lý do khác.

---

## 0. Điều chỉnh phạm vi và công bố xuất xứ hội đồng

Skill này được thiết kế cho bản thảo hoàn chỉnh. Ở đây đối tượng là **một dự án nghiên cứu đang chạy dở**, nên các tiêu chí "Writing Quality", "References", "Citation Format" được đánh `NOT_ASSESSED`. Mọi tiêu chí còn lại áp dụng bình thường, vì câu hỏi của bạn là *đề tài có ổn không*, tức là đúng phần Originality, Methodological Rigor, Evidence Sufficiency, Argument Coherence, Significance.

### Công bố xuất xứ hội đồng (bắt buộc, không được tô hồng)

| Trục | Trạng thái |
|---|---|
| Tách vai (role_separated) | `true` - 5 ghế có nhiệm vụ và tiêu chí riêng |
| Tách ngữ cảnh gọi (fresh_context) | **`false`** - cả 5 ghế chạy tuần tự trong cùng một ngữ cảnh |
| Mù với kết quả ghế khác (blind_to_peer_outputs) | **`false`** |
| Khác dòng model | **`false`** - cùng một model |
| Khác nhà cung cấp | **`false`** |
| Khác người chịu trách nhiệm | **`false`** |

> **Tuyên bố sai số tương quan:** năm ghế này **không độc lập**. Chúng chia sẻ cùng một model, cùng một ngữ cảnh, cùng một thiên kiến. Việc nhiều ghế cùng nêu một vấn đề **không phải** bằng chứng vấn đề đó có thật một cách độc lập. Giá trị thật của báo cáo này nằm ở **các phép kiểm chứng bằng số tôi đã tự chạy trên dữ liệu của bạn** (mục 6), không nằm ở việc có mấy ghế đồng ý.

Toàn bộ kiểm chứng dưới đây tốn **0 USD**, chạy lại được bằng mã trong mục 6.

---

## 1. Phase 0 - Phân tích lĩnh vực và cấu hình hội đồng

| Hạng mục | Kết quả |
|---|---|
| Ngành chính | NLP / đánh giá năng lực suy luận của LLM |
| Ngành phụ | Suy luận nhân quả (Pearl), thống kê thực nghiệm |
| Hệ hình | Thực nghiệm định lượng, thiết kế đối sánh trong cùng đối tượng |
| Loại phương pháp | Benchmark có can thiệp, gây nhiễu có kiểm soát |
| Hạng hội nghị nhắm tới | ACL / EMNLP / NeurIPS D&B track |
| Độ chín | **Kết quả sơ bộ đã có, chưa đủ để nộp** |

**Cấu hình 5 ghế** (bạn có thể yêu cầu đổi):

| Ghế | Danh tính | Trọng tâm |
|---|---|---|
| Journal-Fit | AC của ACL Rolling Review, mảng evaluation và analysis | Độ mới so với văn liệu 2026, mức đóng góp |
| R1 Phương pháp | Nhà thống kê thực nghiệm, chuyên thiết kế paired và cỡ mẫu | Hiệu lực nội tại, ước lượng, ngoại suy |
| R2 Chuyên ngành | Nhà nghiên cứu causal ML, biết CLadder từ gốc | Bao phủ văn liệu, độ chính xác khái niệm |
| R3 Liên ngành | Nhà tâm lý học nhận thức, chuyên thiết kế đối chứng | Giả thuyết cạnh tranh, cấu trúc điều kiện đối chứng |
| Devil's Advocate | Ghế cố định | Phản biện luận điểm lõi |

---

> ## GHI CHÚ VỀ MỤC 2 TỚI MỤC 5
>
> Bốn mục dưới đây là **biên bản vòng 1, giữ nguyên không sửa** để đối chiếu. Chúng được viết trước khi lỗi `C1'` bị phát hiện, nên một số đề xuất trong đó nay đã sai. Đáng chú ý nhất:
>
> - Mục 2 W1 đề xuất đặt tiêu đề là *"cấp đồ thị đúng 100% cũng không khôi phục được năng lực"*. Dữ liệu mới cho thấy **ngược lại**: đồ thị đúng cắt hơn nửa tác hại.
> - Mục 3 W1 đề xuất chạy split anticommonsense làm đối chứng ghép cặp. **Không làm được** - file đó cũng không chứa câu hỏi.
> - Mọi con số trong mục 3 W1 về gán lại trọng số `query_type` và ICC đều dùng ánh xạ item sai.
>
> Đánh giá hiện hành nằm ở **mục 6 tới mục 9**.

## 2. Ghế Journal-Fit - Đánh giá độ phù hợp và độ mới (vòng 1, giữ nguyên)

**Khuyến nghị: Major Revision. Độ tin cậy: 4/5** (mảng evaluation là chuyên môn chính, causal inference là mảng liền kề)

### Tổng quan

Dự án hỏi hai câu: mỗi loại lỗi đồ thị nhân quả đắt bao nhiêu, và năng lực suy luận nhân quả biểu kiến của LLM có thực chất là năng lực từ vựng hay không. Hạ tầng thực nghiệm ở mức tốt hơn phần lớn bài nộp cùng hạng: nhãn chuẩn được tự kiểm chứng bằng giải tích, mọi phép gây nhiễu là tất định, và nhóm tự phát hiện rồi tự rút lại hai khẳng định của chính mình. Đó là dấu hiệu của quy trình có kiểm soát thật.

Vấn đề nằm ở định vị. REPORT.md mục 2.3 xếp Caliper (arXiv:2606.04915) là công trình đi trước cần "định vị khác biệt", nhưng đánh giá thấp mức chồng lấn. Caliper **đã** chạy trên chính tập pseudoword của CLadder và **đã** báo khoảng cách sụp khoảng 19 lần ở đó. Nghĩa là "phát hiện chấn động nhất" của dự án, ở dạng phát biểu hiện tại, **đã được công bố tháng 6/2026**.

Cái còn lại của riêng dự án thì có thật và không tầm thường: Caliper **không** cấp đồ thị đúng cho model, **không** đo chất lượng đồ thị tự dựng, **không** phân rã lỗi theo chiều cạnh. Ba thứ đó là của bạn. Nhưng chúng phải trở thành tiêu đề, chứ không phải phần phụ lục của một phát hiện đã có chủ.

### Điểm mạnh

**S1: Kiểm chứng nhãn chuẩn tới sai số dấu phẩy động.**
Nhóm tự viết solver SCM giải tích và đối chiếu 2.184 model, sai lệch tuyệt đối tối đa 6,66e-16. Rất hiếm bài benchmark nào tự làm việc này.
**Evidence Anchor**: `dataset: results/groundtruth_verification.csv - sai so tuyet doi toi da 6.66e-16 tren 2184 model`

**S2: Hai lần tự rút lại khẳng định của chính mình, có số liệu trước và sau.**
Khẳng định "tái lập với p<0,05" bị rút vì phụ thuộc lần bốc RNG; khẳng định "model yếu không hưởng lợi từ cấu trúc" bị rút vì là giả tạo do cách chấm. Cả hai đều kèm bảng đối chiếu.
**Evidence Anchor**: `text: REPORT.md §6.3 "khang dinh truoc do cua toi rang phat hien loi tai lap voi p<0,05 khong dung vung"`

**S3: Cache toàn bộ nên tái lập bằng 0 đồng.**
**Evidence Anchor**: `dataset: cache/ - khoa SHA256 tren (model, temperature, prompt)`

### Điểm yếu

**W1: Định vị so với Caliper đang đánh giá thấp mức chồng lấn**
**Vấn đề**: REPORT.md §2.3 viết kết quả của dự án "mạnh hơn" Caliper vì "rơi thẳng xuống sàn ngẫu nhiên". Nhưng Caliper đã chạy tập pseudoword của CLadder và ghi nhận khoảng cách sụp ~19 lần cùng hành vi trả lời thoái hoá. Phát biểu "mạnh hơn" không đứng vững ở dạng hiện tại.
**Evidence Anchor**: `text: arXiv:2606.04915 Table 3 "the gap collapses by 17x on CLadder's pseudoword subset"`
**Vì sao quan trọng**: reviewer nào biết Caliper sẽ đọc mục 2 và kết luận đây là kết quả trùng lặp. Đó là lý do reject phổ biến nhất cho bài phân tích.
**Đề xuất**: đổi tiêu đề sang thứ Caliper không có, tức là **điều kiện ORACLE**. Phát biểu đúng và mới là: *cấp đồ thị nhân quả đúng 100 phần trăm không hề khôi phục được năng lực đã mất khi bỏ neo từ vựng.* Caliper chỉ chứng minh model không suy luận cấu trúc khi phải tự dựng; bạn chứng minh nó cũng không suy luận cấu trúc **khi được cho sẵn cấu trúc**. Đó là bước loại trừ mà Caliper thiếu.
**Severity**: Major
**Confidence**: 5 - chuyên môn lõi: định vị đóng góp so với văn liệu

**W2: Khái quát hoá vượt bằng chứng**
**Vấn đề**: WALKTHROUGH.md §5.4 kết luận "LLM không hề có tư duy nhân quả trừu tượng dựa trên cấu trúc logic". Bằng chứng là 3 model, cùng một dòng GPT-4.1, cùng một nhà cung cấp, cùng một thế hệ.
**Evidence Anchor**: `text: WALKTHROUGH.md §5.4 "KET LUAN DANH THEP: LLM khong he co tu duy nhan qua truu tuong"`
**Vì sao quan trọng**: Caliper dùng 9 model từ 3,8B tới 671B mới dám phát biểu ở mức đó. Ba model một dòng không đỡ nổi một mệnh đề phổ quát về "LLM".
**Đề xuất**: hạ phát biểu xuống phạm vi đã đo, hoặc thêm ít nhất một dòng model khác họ.
**Severity**: Major
**Confidence**: 5 - chuyên môn lõi

---

## 3. Ghế R1 - Phản biện phương pháp và thống kê

**Khuyến nghị: Major Revision. Độ tin cậy: 5/5**

### Tổng quan

Thiết kế paired trên nhánh chính là đúng và được thực thi cẩn thận, đặc biệt sau khi sửa lỗi seed. Nhưng ba chỗ trong chuỗi ước lượng chưa chịu được soi kỹ: điểm hoà vốn không tự nhất quán với chính đường hồi quy sinh ra nó, giá của FE về cơ bản chưa nhận dạng được, và **thao tác từ vựng không phải là paired** dù toàn bộ phần còn lại của dự án là paired.

### Điểm mạnh

**S1: Kỷ luật chấm điểm theo câu parse được, kèm parse rate như một đại lượng riêng.**
Đây là confound rất hay bị bỏ qua và nhóm đã xử lý đúng, lại còn dùng nó để tự rút lại một kết luận cũ.
**Evidence Anchor**: `dataset: results/types_parserate.csv - parse rate dao dong 75.9% den 99.3% giua cac dieu kien`

**S2: Cố định seed theo từng (item, loại, k) sau khi đo được phương sai do bốc.**
Phương sai 1,36 pp trung bình, 3,40 pp tối đa, ngang cỡ hiệu ứng đang đo. Đo được rồi mới sửa là cách làm đúng.
**Evidence Anchor**: `text: scripts/pilot.py "Seeding per (item, type, k) keeps each draw independent of what else the run happens to request"`

**S3: Loại trừ được thoái hoá đáp án bằng dữ liệu.**
Tôi đã tự kiểm: tỉ lệ trả lời "yes" trên nhánh pseudoword nằm trong khoảng 40,9 tới 63,2 phần trăm, không có model nào sập về một phía. Caliper báo có model trả "no" 100 phần trăm; ở đây thì không. Nghĩa là mức 50 phần trăm của bạn là đoán mò thật, không phải thoái hoá.
**Evidence Anchor**: `dataset: results/pilot_raw_noncs.csv - ti le pred=='yes' tren cau parse duoc, 12 o dieu kien x model`

### Điểm yếu

**W1: Thao tác từ vựng là between-items, không phải within-item - toàn bộ phần còn lại của dự án là paired**
**Vấn đề**: hai nhánh không dùng chung một item nào. Tôi tái lập phép lấy mẫu: giao của hai tập id là **0/147**. Nhánh commonsense rút từ 26 story, nhánh pseudoword rút từ **10** story (toàn bộ split noncommonsense chỉ có 10 story so với 37). Phân bố query_type lệch: `backadj` 21 so với 29, `ate` 28 so với 20.
**Evidence Anchor**: `dataset: data/test-noncommonsense-v1.5.csv - 0 id trung voi test-commonsense-v1.5.csv, 10 story so voi 37`
**Vì sao quan trọng**: đây chính là trục mà Caliper mạnh hơn bạn. Caliper thay tên biến **ngay trên cùng một câu hỏi**, giữ nguyên đồ thị và bộ số. Bạn so hai tập câu hỏi khác nhau. Reviewer sẽ hỏi ngay: mức rơi là do từ vựng hay do tập câu khác?
**Hai tin tốt tôi đã kiểm giúp:**
- Tôi tính lại độ chính xác nhánh commonsense sau khi **gán lại trọng số theo phân bố query_type của nhánh pseudoword**. Dịch chuyển tối đa **0,98 pp** trên một khoảng cách 28 tới 39 pp. Confound thành phần câu hỏi là có thật nhưng không đáng kể.
- Tôi ước lượng ICC theo story trên nhánh pseudoword: **ICC ~ -0,002, design effect 0,97**. Không có gom cụm theo story, nên n hiệu dụng vẫn là 146 chứ không tụt. Lo ngại về cỡ mẫu hiệu dụng **không thành hiện thực**.
**Đề xuất**: vẫn phải sửa, và có hai cách rẻ:
1. **Split anticommonsense là đối chứng paired hoàn hảo đang nằm sẵn trong `data/` mà chưa dùng.** Tôi đã kiểm: nó dùng **chung đủ 10.392 id** với commonsense, chung 37 story, chung bội số (graph_id, story_id, rung, query_type). McNemar áp dụng trực tiếp được. Nó tách được "từ quen thuộc" khỏi "chiều nhân quả hợp lẽ thường" - đúng chỗ Caliper không tách.
2. Tự thay tên biến ngay trên item commonsense (within-item), bằng chính bộ từ giả. Chi phí 0 đồng ở khâu thiết kế, và đưa thiết kế của bạn ngang Caliper.
**Severity**: Major
**Confidence**: 5 - chuyên môn lõi: thiết kế paired

**W2: Điểm hoà vốn k* không tự nhất quán với đường hồi quy sinh ra nó**
**Vấn đề**: `slope_of` trong `scripts/analyze_types.py:48` dùng `np.polyfit(ks, ys, 1)`, tức OLS **có hệ số chặn tự do**. Nhưng docstring ngay trên đó ghi "a least-squares fit through ORACLE", và công thức hoà vốn `budget / s` với `budget = ORACLE - RAW` **giả định đường thẳng xuất phát đúng tại ORACLE**. Hai giả định này mâu thuẫn nhau.
**Evidence Anchor**: `text: scripts/analyze_types.py:39 "pp lost per erroneous edge, from a least-squares fit through ORACLE" - nhung dong 48 la np.polyfit(ks, ys, 1)`
**Vì sao quan trọng**: giải `a0 + b*k = RAW` cho k* tự nhất quán thì con số đổi đáng kể.

| model | loại | hệ số chặn fit | ORACLE | k* đang báo cáo | k* tự nhất quán |
|---|---|---|---|---|---|
| gpt-4.1-nano | DR | 82,18 | 83,21 | **0,96** | **0,51** |
| gpt-4.1-nano | ED | 81,98 | 83,21 | 0,96 | 0,43 |
| gpt-4.1-mini | DR | 85,93 | 87,77 | 1,43 | 0,88 |
| gpt-4.1 | DR | 88,52 | 89,36 | 1,74 | 1,54 |
| gpt-4.1 | FE | 86,87 | 89,36 | 3,49 | 2,30 |

Với nano, k* rơi từ 0,96 xuống 0,51 - **chênh gần gấp đôi**. Con số đang được báo cáo tới hai chữ số thập phân.
**Đề xuất**: chọn dứt khoát một trong hai - hoặc fit ép qua ORACLE (`slope = -sum(x*(y-orc))/sum(x*x)`), hoặc dùng hệ số chặn fit trong cả tử số lẫn mẫu số. Kèm khoảng tin cậy bootstrap cho k*, `src/stats.py` đã có `bootstrap_break_even`.
**Severity**: Major
**Confidence**: 5 - chuyên môn lõi

**W3: Giá của FE chưa nhận dạng được nhưng vẫn báo cáo như một điểm ước lượng**
**Vấn đề**: FE chỉ có 3 điểm (k=0,1,2) và dữ liệu không đơn điệu. Với gpt-4.1: 89,36 rồi **79,81** rồi **85,19** - tụt xong lại lên. R² của các fit FE là 0,19 (gpt-4.1), 0,52 (mini), 0,50 (nano).
**Evidence Anchor**: `table: results/types_accuracy.csv - gpt-4.1 FE_k1=79.81, FE_k2=85.19, khong don dieu`
**Vì sao quan trọng**: REPORT.md §3 in "FE thừa cạnh 2,09" cạnh "DR 4,20" như hai đại lượng cùng độ tin cậy. Thứ bậc DR > ED > FE - phát biểu như một quy luật trong WALKTHROUGH §5.1 - **chỉ đúng ở mức tương quan thứ hạng trên 3 điểm dữ liệu nhiễu**.
**Đề xuất**: báo cáo FE kèm CI, hoặc nói thẳng chưa nhận dạng được. Không đưa FE vào phát biểu quy luật thứ bậc cho tới khi có k=3 và n lớn hơn.
**Severity**: Major
**Confidence**: 5 - chuyên môn lõi

**W4: Ngoại suy tuyến tính trong khi đường cong đang bão hoà**
**Vấn đề**: DR của gpt-4.1: 89,36 / 84,51 / 77,24 / 77,78. Ba điểm cuối cho thấy đường cong **phẳng ra** chứ không tuyến tính. Fit một đường thẳng rồi ngoại suy để tìm giao điểm với sàn là giả định tuyến tính ngoài vùng đã đo.
**Evidence Anchor**: `table: results/types_accuracy.csv - gpt-4.1 DR_k2=77.24, DR_k3=77.78`
**Vì sao quan trọng**: nếu bão hoà, k* thực tế **xa hơn** con số ngoại suy, và luận điểm "chỉ 1 tới 2 cạnh sai là mất sạch lợi ích" bị nới lỏng.
**Đề xuất**: hoặc thêm dạng hàm bão hoà, hoặc chỉ phát biểu k* trong khoảng đã đo và nói rõ đó là chặn dưới.
**Severity**: Major
**Confidence**: 5 - chuyên môn lõi

### Câu hỏi cho tác giả

1. Tại sao chọn split noncommonsense có sẵn (10 story, không trùng item) thay vì split anticommonsense (10.392 id trùng khớp hoàn toàn, paired được)?
2. Con số 5,24 / 4,41 / 6,50 ở dự đoán cộng tính dùng giá lỗi nào - giá từ `np.polyfit` hay giá ép qua ORACLE? Nếu là cái trước thì dự đoán và ngân sách đang dùng hai gốc toạ độ khác nhau.
3. Độ khớp "dưới 0,3 pp" ở nano và mini có kèm khoảng tin cậy không? Với n=147, sai số chuẩn của một hiệu 5 pp đã vào khoảng 4 pp - độ khớp 0,27 pp có thể là trùng hợp.

---

## 4. Ghế R2 - Phản biện chuyên ngành và văn liệu

**Khuyến nghị: Major Revision. Độ tin cậy: 4/5**

### Điểm mạnh

**S1: Lựa chọn CLadder thay NoisyCausal được biện minh bằng lập luận phương pháp, không phải bằng tiện lợi.**
Bốn lý do trong WALKTHROUGH §2 đều đứng vững, đặc biệt việc chỉ ra NoisyCausal tiêm confounder nhưng vẫn chấm theo SCM sạch. Việc `src/noise.py` tách `answer_preserving` là xử lý đúng nghịch lý đó.
**Evidence Anchor**: `text: WALKTHROUGH.md §2 "NoisyCausal tiem bien an gay nhieu lam xuat hien duong backdoor nhung lai cham theo clean SCM"`

### Điểm yếu

**W1: So sánh F1 xuyên bộ dữ liệu - đúng loại lỗi mà dự án đã tự rút lại một lần**
**Vấn đề**: REPORT.md §4 viết F1 0,52-0,63 "thấp hơn hẳn con số 85,2 NoisyCausal báo cho GPT-4". Tôi đã tra bản PDF gốc: 85,2 là dòng "GPT-4 (clean, structured prompt)" ở **Table 3**, đo trên **bộ dữ liệu của chính NoisyCausal**, không phải CLadder. Đồ thị khác, prompt khác, và bài báo không nói F1 của họ có phân biệt chiều cạnh hay không - trong khi F1 của bạn thì có.
**Evidence Anchor**: `text: 2605.04313v1.pdf Table 3 "GPT-4 (clean, structured prompt) 87.1 83.5 85.2"`
**Vì sao quan trọng**: đây đúng là loại lỗi bắc cầu giữa hai bảng khác tổng thể mà dự án đã phải rút lại một lần (con số "ngân sách sai số 19,68 pp"). Lặp lại nó ở mục 4 sẽ làm mất uy tín cả những chỗ làm đúng.
**Đề xuất**: bỏ so sánh, hoặc giữ nhưng ghi rõ ba khác biệt (bộ dữ liệu, cỡ đồ thị, định nghĩa F1 có hay không phân biệt chiều).
**Severity**: Major
**Confidence**: 5 - đã tra trực tiếp bản PDF trong repo

**W2: Thiếu đối chứng phân ly giữa "từ quen thuộc" và "chiều nhân quả hợp lẽ thường"**
**Vấn đề**: pseudoword bỏ đồng thời **hai** thứ - tính quen thuộc của từ, và tri thức về chiều nhân quả đúng. Không có điều kiện nào tách được hai cái.
**Evidence Anchor**: `absence: thiet ke thuc nghiem - can dieu kien tu that nhung chieu nhan qua sai; da kiem REPORT.md §2, WALKTHROUGH.md §4.2, scripts/pilot.py`
**Vì sao quan trọng**: cơ chế mà dự án đề xuất ở REPORT.md §2.2 là "tri thức ngữ nghĩa giữ cho model không nhầm chiều". Muốn chứng minh cơ chế đó thì phải có điều kiện giữ từ thật nhưng **đảo chiều tri thức**. Split anticommonsense làm đúng việc này và đang nằm sẵn trong `data/`, paired hoàn hảo.
**Đề xuất**: chạy nhánh anticommonsense. Đây là đóng góp phân ly mà Caliper không có, và nó rẻ.
**Severity**: Major
**Confidence**: 4 - chuyên môn liền kề: thiết kế đối chứng nhận thức

---

## 5. Ghế R3 - Góc nhìn liên ngành

**Khuyến nghị: Major Revision. Độ tin cậy: 3/5** (tâm lý học nhận thức là chuyên môn chính, thống kê LLM là liền kề)

### Điểm yếu

**W1: "Đoán mò" và "không có năng lực" không phải một mệnh đề**
**Vấn đề**: từ chỗ độ chính xác bằng 50 phần trăm, REPORT.md §2 suy ra model "không có tư duy nhân quả trừu tượng". Nhưng 50 phần trăm cũng tương thích với nhiều cơ chế khác: mất khả năng gắn ký hiệu với đối tượng, quá tải bộ nhớ làm việc khi phải giữ 3-5 từ vô nghĩa, hoặc mất khả năng phân biệt hai từ giả có hình thái giống nhau (`zuph` với `xevu` với `uvzi`).
**Evidence Anchor**: `text: data/test-noncommonsense-v1.5.csv "Vubr has a direct effect on zuph and uvzi. Wibl has a direct effect on zuph."`
**Vì sao quan trọng**: trong tâm lý học nhận thức đây là phân biệt kinh điển giữa *competence* và *performance*. Một người biết đại số vẫn làm sai nếu bạn đặt tên biến là `zuph` và `uvzi` rồi bắt giữ bốn cái trong đầu. Đó không phải bằng chứng người đó không biết đại số.
**Đề xuất**: thêm một điều kiện dùng ký hiệu **ngắn và dễ phân biệt** (`A`, `B`, `C`) thay vì từ giả nhiều âm tiết. Nếu năng lực phục hồi thì nguyên nhân là gắn ký hiệu, không phải cấu trúc nhân quả. Đây là thí nghiệm rẻ và có sức phân ly rất cao.
**Severity**: Major
**Confidence**: 4 - chuyên môn lõi: thiết kế đối chứng nhận thức

**W2: Giai đoạn 3 đã chuyển hướng dựa trên một tương quan chưa được khảo sát đúng cách**
**Vấn đề**: kết luận "cổng theo F1 không thể có lãi" dựa trên `corr(F1, correct) ~ 0`. Nhưng tương quan Pearson giữa một biến bị chặn khoảng và một biến nhị phân, trên một mẫu mà **phương sai của F1 rất hẹp** (model gần như luôn thiếu cạnh, gần như không bao giờ đảo chiều), thì gần 0 là điều được kỳ vọng ngay cả khi có quan hệ thật.
**Evidence Anchor**: `table: results/induction_quality.csv - canh dao chieu chi 0.014 den 0.048 moi item tren commonsense`
**Vì sao quan trọng**: dự án đã **khai tử** cả một giai đoạn dựa trên phép kiểm này. Nếu phép kiểm không có sức mạnh thì việc khai tử là quá sớm.
**Đề xuất**: kiểm lại trên nhánh pseudoword, nơi lỗi đảo chiều tăng 4 tới 10 lần và phương sai F1 rộng ra. Nếu tương quan xuất hiện ở đó thì kết luận đúng phải là "cổng theo F1 vô dụng khi có neo ngữ nghĩa, hữu dụng khi không có" - một phát biểu **mạnh hơn nhiều** so với "cổng theo F1 vô dụng".
**Severity**: Major
**Confidence**: 4 - chuyên môn lõi: đo lường tương quan trên biến bị chặn

---

## 6. Ghế Devil's Advocate - vòng 2

### Calibration Status

`NOT_CALIBRATED`

### Phản biện mạnh nhất

Vòng 1 tôi tấn công nhầm chỗ. Tôi lập luận rằng mức rơi tập trung ở loại câu thuần số học nên thứ bị phá huỷ không phải năng lực nhân quả. Lập luận đó dựa trên một ánh xạ item sai, và quan trọng hơn, nó chấp nhận **tiền đề sai của chính bài**: rằng có một mức rơi cần giải thích.

Không có mức rơi nào cả. Có một prompt không chứa câu hỏi.

Đó mới là phản biện mạnh nhất, và nó nhắm vào tầng dưới tất cả các tầng khác: **không tài liệu nào trong dự án ghi lại việc đã đọc một prompt hoàn chỉnh của nhánh pseudoword.** Có kiểm chứng nhãn chuẩn tới 6,66e-16, có verify 3.000/3.000 khớp số cạnh, có kiểm parse rate, có kiểm cân bằng đáp án, có kiểm thiên vị trả lời một phía. Cả một bộ kiểm chứng công phu, và không cái nào bắt được lỗi vì tất cả đều kiểm **những gì chương trình tính**, không cái nào kiểm **những gì model thực sự nhìn thấy**.

Bài học này lớn hơn mọi kết quả trong bài, và nó nên được viết vào bài chứ không chỉ vào sổ tay: một benchmark nhiều thành phần có thể im lặng trả về một phần của prompt, và mọi phép kiểm thống kê hạ nguồn vẫn chạy trơn tru trên phần thiếu đó.

Về kết quả mới ở vòng 2, tôi vẫn còn hai chỗ để tấn công, nêu ở `M1` và `M2`.

### Danh sách vấn đề

#### CRITICAL

| # | Chiều | Mô tả | Evidence Anchor | Confidence |
|---|---|---|---|---|
| C1' | Nền móng sụp | Toàn bộ phát hiện tiêu đề của vòng 1 dựa trên các file không chứa câu hỏi (0,00% prompt có `?`), nên mức 50% là tất yếu chứ không phải kết quả. **Đã được tác giả xác nhận, rút, và chặn bằng guard trong `make_items`.** | `dataset: test-noncommonsense-v1.5.csv - 0.00% prompt chua dau '?', doi chieu full_v1.5_default.csv 100%` | 5 - kiểm trực tiếp trên cả bốn file |

**Trạng thái:** đã giải quyết. Guard đã có, file kết quả hỏng đã đổi tên tiền tố `INVALID_no_question_`, hai tài liệu đã đính chính ở đầu.

#### MAJOR

| # | Chiều | Mô tả | Evidence Anchor | Confidence |
|---|---|---|---|---|
| M1 | Khái quát hoá quá mức | Luận điểm mới - *neo từ vựng giúp TRÍCH XUẤT đồ thị chứ không giúp SUY LUẬN trên đồ thị* - ghép hai phép đo có hiệu lực khác nhau. Phần TRÍCH XUẤT (mục 9 REPORT) là so sánh **between-items** giữa 90 item từ thật và 57 item từ giả trong cùng nhánh. Phần SUY LUẬN là ghép cặp. Chỉ nửa sau chịu được McNemar. | `dataset: results/induction_raw.csv - 90 item story tu that so voi 57 item story nonsense, khong ghep cap` | 5 - chuyên môn lõi |
| M2 | Đứt chuỗi logic | Kết luận mục 7 REPORT - *cái bị mất là tri thức về thế giới, không phải khả năng gắn ký hiệu* - suy ra từ việc 0/9 ô `PSEUDO - SYMBOL` đạt p<0,05. Đó là **luận cứ từ việc không bác bỏ được**. Với n=174 và hiệu ứng cỡ 3-6 pp, phép kiểm này không đủ sức mạnh để phân biệt "không có khác biệt" với "chưa phát hiện được khác biệt". | `table: results/lexical_mcnemar.csv - PSEUDO-SYMBOL, p tu 0.053 den 1.00, khong o nao dat p<0.05` | 5 - chuyên môn lõi |
| M3 | Bằng chứng chưa đủ | `gpt-4.1` với `SYMBOL` ở điều kiện `ORACLE` vẫn giữ nguyên tác hại (-8,33 pp, p=0,0243), trong khi 5 ô còn lại đều hồi phục. REPORT gọi đây là "ngoại lệ trung thực, chưa giải thích được". Một ngoại lệ ở đúng model mạnh nhất, ở đúng điều kiện làm nên luận điểm chính, cần nhiều hơn một dòng ghi chú. | `table: results/lexical_mcnemar.csv - gpt-4.1 ORACLE SYMBOL-KEEP delta=-8.33 p=0.0243` | 4 |

#### MINOR

| # | Chiều | Mô tả | Evidence Anchor | Confidence |
|---|---|---|---|---|
| m1 | Khoảng trống bằng chứng | Giả thuyết "bỏ sót có chọn lọc" giải thích độ lệch của `gpt-4.1` đã được rút cùng mô hình cộng tính, nhưng vẫn còn nằm trong mục "chưa làm" của REPORT như một việc cần kiểm. Cơ sở để kiểm nó đã không còn. | `text: REPORT.md muc 11 "Kiem gia thuyet bo sot co chon loc"` | 3 |

### Giải thích thay thế bị bỏ qua

1. **Độ dài và cách tách token.** `SYMBOL` làm prompt ngắn đi rõ rệt, `PSEUDO` giữ độ dài gần với `KEEP`. Không điều kiện nào kiểm soát việc này, nên một phần của khoảng cách `SYMBOL - KEEP` có thể là hiệu ứng độ dài chứ không phải hiệu ứng tri thức.
2. **`ORACLE` làm dễ bài toán chứ không phải bù tri thức.** Khối cấu trúc nêu lại tên biến ngay trước câu hỏi, nên nó cũng là một bản tóm tắt vị trí gần. Một điều kiện `ORACLE` với đồ thị **sai hoàn toàn nhưng cùng độ dài** sẽ tách được hai thứ này. Dự án có sẵn `DR_k1` nhưng đó là đồ thị sai một cạnh, chưa phải đối chứng độ dài thuần.

### Quan sát không phải lỗi

- Việc giữ lại file hỏng với tiền tố `INVALID_no_question_` kèm README giải thích, thay vì xoá, là cách xử lý đúng.
- Guard trong `make_items` báo lỗi bằng tiếng Việt và chỉ thẳng cách làm đúng thay vì chỉ raise. Đó là thiết kế lỗi tốt.

---

## 7. Quyết định biên tập - vòng 2

### MINOR REVISION

Vòng 1 là `MAJOR REVISION` với ba vấn đề chặn. Cả ba đã được xử lý, và vấn đề nghiêm trọng hơn cả ba - `C1'` - cũng đã được tìm ra và sửa tận gốc trong cùng lượt.

| Vấn đề chặn vòng 1 | Trạng thái |
|---|---|
| B1 - kết luận tiêu đề không suy ra được từ dữ liệu | **Đã giải quyết.** Rút toàn bộ, thay bằng luận điểm mới có 14/36 phép kiểm p<0,05 |
| B2 - thao tác từ vựng là between-items | **Đã giải quyết.** `src/lexical.py` đổi từ vựng trong cùng item, McNemar áp dụng trực tiếp |
| B3 - chồng lấn với Caliper chưa thừa nhận đúng | **Đã giải quyết.** REPORT mục 12 và WALKTHROUGH 8.3 đặt Caliper là tiền đề, liệt kê ba đóng góp riêng |
| R4 - k\* không tự nhất quán | **Đã giải quyết.** Giải từ chính đường fit, thêm CI bootstrap |
| R5 - FE trong quy luật thứ bậc | **Đã giải quyết.** 6/9 giá bị đánh dấu CI chứa 0, thứ bậc đã rút |

### Vì sao không phải Accept

Ba vấn đề `MAJOR` mới ở mục 6 đều là vấn đề **diễn giải**, không phải vấn đề dữ liệu, và cả ba sửa được bằng cách viết lại phát biểu hoặc chạy thêm một điều kiện rẻ. Không cái nào đe doạ kết quả lõi.

### Các mục sửa bắt buộc - vòng 2

**R1: Tách phát biểu "TRÍCH XUẤT so với SUY LUẬN" thành hai mức độ tin cậy khác nhau**

- **Vấn đề**: nửa "trích xuất" là between-items, nửa "suy luận" là ghép cặp (DA `M1`).
- **Yêu cầu**: hoặc chạy `induction.py` trên cả ba bộ từ vựng ghép cặp, hoặc ghi rõ trong chính câu phát biểu rằng nửa đầu là bằng chứng yếu hơn.
- **Chi phí**: khoảng 1,5 USD nếu chạy; 0 USD nếu chỉ ghi chú.
- **Tiêu chí nghiệm thu**: không còn câu nào ghép hai phép đo khác hiệu lực thành một mệnh đề duy nhất mà không phân biệt.

**R2: Không phát biểu khẳng định từ việc không bác bỏ được**

- **Vấn đề**: "cái bị mất là tri thức, không phải gắn ký hiệu" dựa trên 0/9 ô không đạt p<0,05 (DA `M2`).
- **Yêu cầu**: đổi thành phát biểu về giới hạn phát hiện - *ở n=174, không phát hiện được khác biệt giữa `SYMBOL` và `PSEUDO`; nếu có, nó nhỏ hơn khoảng X pp* - kèm phân tích power tối thiểu. `src/stats.py` đã có `mcnemar_power`.
- **Chi phí**: 0 USD.
- **Tiêu chí nghiệm thu**: mục 7 REPORT nêu độ lớn hiệu ứng nhỏ nhất mà thiết kế phát hiện được.

**R3: Xử lý ngoại lệ `gpt-4.1` với `SYMBOL`**

- **Vấn đề**: ô duy nhất không hồi phục nằm ở model mạnh nhất, đúng điều kiện làm nên luận điểm (DA `M3`).
- **Yêu cầu**: nêu ít nhất một giả thuyết kiểm được, hoặc gộp vào phân tích power ở R2 nếu đó chỉ là dao động mẫu.
- **Chi phí**: 0 USD.
- **Tiêu chí nghiệm thu**: ngoại lệ được đối xử như một quan sát cần giải thích, không phải một dòng ghi chú.

**R4: Thêm đối chứng độ dài prompt**

- **Vấn đề**: `SYMBOL` làm prompt ngắn hẳn, không có điều kiện nào tách hiệu ứng độ dài khỏi hiệu ứng tri thức.
- **Yêu cầu**: thêm một bộ từ vựng gồm **từ thật nhưng không liên quan** (ví dụ `banana`, `guitar`, `cloud`) - giữ độ dài và tính "là từ có nghĩa", chỉ bỏ quan hệ nhân quả hợp lẽ thường. Đây cũng chính là đối chứng anticommonsense mà vòng 1 đề xuất, nhưng làm được vì không phụ thuộc file hỏng.
- **Chi phí**: khoảng 2,3 USD cho một bộ từ vựng nữa.
- **Tiêu chí nghiệm thu**: có bốn bộ `KEEP` / `IRRELEVANT` / `SYMBOL` / `PSEUDO` trên cùng item.

**R5: Đưa lỗi `C1'` vào bài báo như một đóng góp về phương pháp**

- **Vấn đề**: hiện nó chỉ là một đính chính nội bộ.
- **Yêu cầu**: một đoạn ngắn trong phần Limitations hoặc một khối cảnh báo - *split `test-*` của CLadder v1.5 không chứa câu hỏi; chạy trên chúng cho ra đúng 50% và trông y hệt một phát hiện*. Đây là thông tin có ích thật cho cộng đồng dùng CLadder.
- **Chi phí**: 0 USD.
- **Tiêu chí nghiệm thu**: bài báo có cảnh báo này kèm số liệu 0,00% so với 100%.

### Lộ trình sửa, theo thứ tự truy vết nguồn

- [ ] R1 - `must_fix` - tách hai mức độ tin cậy của luận điểm trích xuất so với suy luận
- [ ] R2 - `must_fix` - thay khẳng định bằng giới hạn phát hiện, kèm power
- [ ] R3 - `must_fix` - xử lý ngoại lệ gpt-4.1 SYMBOL ORACLE
- [ ] R4 - `must_fix` - thêm bộ từ vựng `IRRELEVANT` làm đối chứng độ dài
- [ ] R5 - `must_fix` - đưa cảnh báo về split `test-*` vào bài báo
- [ ] S1 - `should_fix` - bỏ "bỏ sót có chọn lọc" khỏi danh sách việc cần làm
- [ ] S2 - `should_fix` - thêm một dòng model khác họ
- [ ] S3 - `consider` - kiểm lại `corr(F1, correct)` dưới `PSEUDO`, nơi phương sai F1 rộng hơn

---

## 8. Đánh giá lại các ghế

| Ghế | Vòng 1 | Vòng 2 | Ghi chú |
|---|---|---|---|
| Journal-Fit | Major Revision | **Minor Revision** | Định vị so với Caliper đã sửa đúng; đóng góp riêng giờ rõ ràng |
| R1 Phương pháp | Major Revision | **Minor Revision** | Thiết kế ghép cặp đã có; CI và guard đã có |
| R2 Chuyên ngành | Major Revision | **Minor Revision** | So sánh F1 xuyên bộ dữ liệu đã bỏ; đối chứng phân ly vẫn thiếu (R4) |
| R3 Liên ngành | Major Revision | **Minor Revision** | Giả thuyết gắn ký hiệu đã được kiểm bằng `SYMBOL`, dù kết luận cần phát biểu lại (R2) |
| Devil's Advocate | 1 CRITICAL, 3 MAJOR | 1 CRITICAL đã giải quyết, 3 MAJOR mới | Xem mục 6 |

Nhắc lại cảnh báo ở mục 0: năm ghế này chạy trong cùng một ngữ cảnh, cùng một model, **không độc lập**. Giá trị của báo cáo nằm ở các phép kiểm chạy trên dữ liệu, không ở số ghế đồng ý. Vòng 1 là minh hoạ trực tiếp: cả năm ghế đều bỏ sót lỗi nghiêm trọng nhất, và nó chỉ lộ ra khi có người mở một prompt hoàn chỉnh ra đọc.

---

## 9. Kết luận cho câu hỏi "đề tài có ổn không"

**Đề tài ổn hơn trước, và lý do là nó vừa sống sót qua một lỗi đủ sức giết nó.**

Phát hiện tiêu đề cũ không còn. Thay vào đó là một luận điểm hẹp hơn nhưng đứng vững:

> Cấp đồ thị nhân quả đúng cắt hơn nửa tác hại của việc ẩn danh tên biến. Không đồ thị: 6/6 phép so sánh đạt p<0,05, hại trung bình 11,15 pp. Có đồ thị đúng: 1/6 và 5,20 pp.

Ba lý do luận điểm này tốt hơn cái cũ:

1. **Ghép cặp.** Cùng 174 item, chỉ đổi tên biến. McNemar áp dụng trực tiếp, không phải so hai tập câu hỏi khác nhau.
2. **Tái lập được Caliper.** Mức tụt 10-13 pp nằm trong khoảng 7,6-29,6 pp mà Caliper báo, bằng một thiết kế độc lập. Kết quả cũ - rơi thẳng xuống 50% - thì không tái lập được cái gì cả.
3. **Nằm ở chỗ Caliper không có.** Caliper không bao giờ cấp đồ thị cho model. Điều kiện `ORACLE` là của riêng dự án, và nó là chỗ luận điểm sống.

Bốn trong năm mục sửa vòng 2 tốn 0 USD. Mục còn lại khoảng 2,3 USD.


---

## 10. Trạng thái các mục sửa vòng 2 (cập nhật 2026-09-08)

Tác giả đã bổ sung bộ từ vựng thứ tư, `PERMUTE`, hoán vị chính tên biến của item sang vị trí khác trong đồ thị của chính nó bằng một derangement. Điều này **đóng cùng lúc hai mục sửa**.

| Mục | Nội dung | Trạng thái |
|---|---|---|
| R1 | Tách hai mức tin cậy của luận điểm trích xuất so với suy luận | **Đã ghi chú.** WALKTHROUGH 5.7 nêu rõ nửa "trích xuất" là between-items. Chạy induction ghép cặp vẫn còn để mở |
| R2 | Không phát biểu khẳng định từ việc không bác bỏ được | **Đã giải quyết** |
| R3 | Xử lý ngoại lệ `gpt-4.1` với `SYMBOL` ở `ORACLE` | **Đã giải quyết một phần** |
| R4 | Thêm đối chứng độ dài prompt | **Đã giải quyết** |
| R5 | Đưa cảnh báo về split `test-*` vào bài báo | Còn để mở, tuỳ bản thảo |

### Vì sao R2 đã đóng

Vòng 2 tôi phản đối việc suy ra *"cái bị mất là tri thức, không phải gắn ký hiệu"* từ 0/9 ô không đạt p<0,05 - đó là luận cứ từ việc không bác bỏ được, và ở n=174 phép kiểm không đủ sức phân biệt "không có khác biệt" với "chưa phát hiện được".

`PERMUTE` biến nó thành một lập luận có **đối chứng dương**:

| Bậc | Chênh TB | p<0,05 |
|---|---|---|
| `PERMUTE` - `KEEP` | **-7,52 pp** | **5/12** |
| `SYMBOL` - `PERMUTE` | +0,73 pp | 0/12 |
| `PSEUDO` - `SYMBOL` | +0,13 pp | 2/12 |

Cùng thiết kế, cùng n, cùng số phép kiểm: bậc một **phát hiện được** hiệu ứng 7,52 pp. Vậy hai bậc sau không phát hiện được gì là bằng chứng về độ lớn hiệu ứng, không phải chuyện thiếu power. Đây là cách đúng để phát biểu một kết quả null.

### Vì sao R4 đã đóng

Tôi yêu cầu một bộ từ vựng gồm từ thật không liên quan, để giữ độ dài và tính "là từ có nghĩa". `PERMUTE` làm tốt hơn đề xuất của tôi: nó không lấy từ bên ngoài mà **dùng lại chính từ vựng của item**, nên bộ từ giống hệt đến từng chữ cái, chỉ khác ở việc tên nào ứng với vị trí nào.

Lệch độ dài so với `KEEP`: `PERMUTE` **+9 ký tự**, `SYMBOL` -169, `PSEUDO` -127. Confound độ dài bị loại sạch, và nó cho thấy độ dài không phải nguyên nhân - bộ dài nhất lại là bộ mất nhiều nhất.

### R3 giải quyết được một phần

`PERMUTE` cho biết vì sao có ô đồ thị không cứu được: nó là dạng nhiễu **đối kháng** duy nhất trong bốn bộ, vì nó cấp cho model một prior sai lệch cạnh tranh trực tiếp với đồ thị. Đúng như vậy, `PERMUTE` là bộ có tỉ lệ ô còn ý nghĩa ở `ORACLE` cao nhất (2/3). Nhưng ngoại lệ ban đầu là ở `SYMBOL`, không phải `PERMUTE`, nên lời giải thích này chưa phủ hết. Vẫn cần phân tích power như R2.

### Phát hiện mới đáng chú ý

`gpt-4.1-nano` có `Delta_struct` âm mạnh nhất đúng ở `PERMUTE` (-9,21 pp). Model yếu nhất bị hại nhiều nhất khi phải phân xử giữa một prior sai lệch và một đồ thị đúng. Đây là bằng chứng trực tiếp nhất cho luận điểm gốc của đề tài - *cấu trúc không miễn phí* - và nó xứng đáng là một mục riêng trong bài, không phải một dòng trong bảng.

### Đánh giá lại

Không có finding CRITICAL hay MAJOR nào mới. Hai trong ba MAJOR của vòng 2 đã đóng. **Quyết định giữ nguyên MINOR REVISION**, nhưng khoảng cách tới Accept đã ngắn hẳn: việc còn lại chủ yếu là chạy induction ghép cặp và viết lại phát biểu, không phải sửa thiết kế.


---

## 11. Vòng 3 - thẩm định lại sau khi chạy induction ghép cặp (2026-09-08)

**Chế độ `re-review`. Quyết định: MINOR REVISION.** Không có finding CRITICAL. Hai finding MAJOR mới, cả hai đều là lỗi **phát biểu**, sửa bằng cách viết lại và bổ sung đường sàn - không phải sửa thiết kế, không phải chạy lại.

### 11.1 Xác minh mục sửa R1

| | |
|---|---|
| Yêu cầu vòng 2 | Nửa "trích xuất" là between-items (90 với 57 item), phải làm ghép cặp hoặc ghi rõ là bằng chứng yếu hơn |
| Tác giả đã làm | Chạy `induction.py` trên cả bốn bộ từ vựng, cùng 174 item, thêm `--lexicon` và `--drop-nonsense` |
| **Kết luận** | **ĐÓNG.** Wilcoxon ghép cặp trên F1 từng item, 8/9 ô đạt p<0,05. Đây là ghép cặp thật, không phải ghép cặp trên danh nghĩa |

Trạng thái năm mục vòng 2: R1 **đóng**, R2 **đóng**, R4 **đóng**, R3 đóng một phần, R5 còn để mở (tuỳ bản thảo).

### 11.2 Ba kiểm chứng tôi tự chạy trên dữ liệu

Đều 0 USD, mã ở `scripts/induction_baselines.py`.

**Kiểm chứng 1 - hiệu ứng có phải do model xuất nhiều cạnh hơn không? KHÔNG.**

Số cạnh model xuất ra gần như phẳng giữa các bộ (1,75 tới 2,27). Chuẩn hoá lại theo số cạnh xuất ra thì phép phân ly còn nguyên:

| Tình huống | Đảo chiều trên mỗi cạnh xuất ra |
|---|---|
| Prior đúng (`KEEP`) | 0,022 - 0,039 |
| Prior vắng mặt (`SYMBOL`, `PSEUDO`) | 0,047 - 0,075 |
| Prior sai (`PERMUTE`) | **0,172 - 0,233** |

**Kiểm chứng 2 - phép phân ly có ý nghĩa thống kê không? CÓ, rất mạnh.**

Wilcoxon ghép cặp trên số cạnh đảo chiều mỗi item, `PERMUTE` so với từng bộ prior-vắng-mặt: **6/6 ô đạt p<0,0001**. Gộp `SYMBOL`+`PSEUDO` thành một nhóm: 3/3 với p từ 4e-6 tới 2e-7. Ở mức item, tỉ lệ có ít nhất một cạnh đảo chiều là 24,7-39,1% dưới `PERMUTE`, so với 8,0-13,8% khi prior vắng mặt và 4,6-6,9% khi prior đúng.

Phép phân ly lõi **đứng vững dưới cả ba cách nhìn**: giá trị tuyệt đối, chuẩn hoá, và mức item.

**Kiểm chứng 3 - hai đường sàn mô phỏng.** Đây là chỗ tìm ra vấn đề.

| Tác nhân giả định | Đảo chiều | F1 |
|---|---|---|
| Đoán ngẫu nhiên (cùng số cạnh, bốc đều trên các cặp nút) | 1,345 | **0,362** |
| Chỉ dùng tri thức thế giới (xuất đồ thị `KEEP`, chấm theo `PERMUTE`) | **0,966** | 0,181 |
| *Model thật dưới `PERMUTE`* | *0,316 - 0,529* | *0,384 - 0,424* |

### 11.3 Finding MAJOR

| # | Chiều | Mô tả | Evidence Anchor | Confidence |
|---|---|---|---|---|
| N1 | Bằng chứng chưa đủ | **F1 được báo cáo không kèm sàn ngẫu nhiên.** Trên đồ thị 3-5 nút chỉ có 6 tới 20 cặp có hướng, nên đoán mò đã đạt F1 = 0,362. `PERMUTE` cho F1 0,384-0,424, tức chỉ **hơn sàn 0,022 tới 0,062**. Người đọc thấy "F1 = 0,406" sẽ hiểu là năng lực trung bình, thực tế nó gần như bằng đoán mò. `KEEP` (0,462-0,608) mới thực sự vượt sàn. | `table: results/induction_baselines.csv - f1_tren_san_ngau_nhien, gpt-4.1 PERMUTE = 0.043` | 5 - chuyên môn lõi: diễn giải thước đo |
| N2 | Khái quát hoá quá mức | **Phát biểu "model lấy chiều của cạnh từ tri thức về thế giới, không phải từ đề bài" quá tuyệt đối.** Một tác nhân bỏ hẳn văn bản và chỉ trả lời theo tri thức sẽ đạt 0,966 cạnh đảo chiều. Model thật đạt 0,316-0,529, tức chỉ đi được **32,7% tới 54,8%** quãng đường tới tác nhân đó. Chúng đọc văn bản **một phần**, không phải bỏ qua văn bản. | `table: results/induction_baselines.csv - phan_tram_duong_toi_tri_thuc, 32.7 den 54.8` | 5 - chuyên môn lõi |

### 11.4 Finding MINOR

| # | Chiều | Mô tả | Evidence Anchor | Confidence |
|---|---|---|---|---|
| n1 | Bất thường chưa xử lý | `gpt-4.1-nano` với `SYMBOL` có F1 **cao hơn** `KEEP` (+0,046, p=0,0314) - ngược chiều toàn bộ luận điểm, và có ý nghĩa thống kê. Báo cáo không nhắc tới. Có thể chỉ là một trong 9 ô ở mức alpha 0,05, nhưng phải nói ra chứ không im lặng. | `table: results/lexical_induction.csv - gpt-4.1-nano SYMBOL f1_vs_KEEP=+0.046 p=0.0314` | 4 |

### 11.5 Điểm mạnh mới

**S1: Phép phân ly prior-sai so với prior-vắng-mặt là một thiết kế thật sự tốt.**
Nó tách được hai thứ mà gần như mọi công trình về ẩn danh biến đều gộp làm một. Caliper không có nó. Cách phát biểu cần sửa, nhưng **thiết kế thì đúng và phát hiện thì có thật.**
**Evidence Anchor**: `dataset: results/induction_raw_lexPERMUTE.csv doi chieu lexSYMBOL/lexPSEUDO - 6/6 Wilcoxon p<0.0001`

**S2: Con số bị rút trước đây quay lại sau khi đo đúng cách.**
"Đảo chiều tăng 4-10 lần" từng bị rút vì đo trên dữ liệu hỏng. Đo lại ghép cặp trên prompt đầy đủ cho 5,1-7,7 lần. Hướng đúng, và giờ có cơ sở. Việc tác giả rút nó trước rồi mới lấy lại bằng bằng chứng sạch là quy trình đúng.
**Evidence Anchor**: `text: REPORT.md muc 9 "Huong thi dung, nhung no duoc do tren du lieu hong. Gio no duoc do lai dung cach"`

### 11.6 Các mục sửa bắt buộc - vòng 3

**R1: Báo cáo F1 kèm sàn ngẫu nhiên ở mọi chỗ F1 xuất hiện**

- **Yêu cầu**: thêm dòng sàn (0,362) vào bảng mục 9 REPORT và 5.7 WALKTHROUGH, hoặc đổi sang báo cáo F1 vượt sàn. Nói rõ `PERMUTE` gần như bằng đoán mò.
- **Chi phí**: 0 USD, `scripts/induction_baselines.py` đã tính sẵn.
- **Nghiệm thu**: không còn con số F1 nào đứng một mình.

**R2: Hạ phát biểu cơ chế từ tuyệt đối xuống mức độ**

- **Yêu cầu**: thay *"model lấy chiều của cạnh từ tri thức, không phải từ đề bài"* bằng *"khi tên biến gợi một chiều nhân quả sai, model đi theo tri thức ở khoảng một phần ba tới một nửa quãng đường, thay vì đọc chiều đã được nêu trong đề bài"*. Kèm con số 32,7-54,8%.
- **Chi phí**: 0 USD.
- **Nghiệm thu**: mọi phát biểu cơ chế đều có định lượng mức độ, không còn dạng "không phải X mà là Y".

**R3: Nêu bất thường `nano` với `SYMBOL`**

- **Chi phí**: 0 USD.
- **Nghiệm thu**: bất thường được nêu, kèm nhận định đó là nhiễu hay hiện tượng thật.

### 11.7 Trạng thái sau khi tác giả sửa ngay trong lượt

| Mục | Trạng thái |
|---|---|
| R1 - báo cáo F1 kèm sàn ngẫu nhiên | **Đã sửa.** Khối sàn 0,362 thêm vào REPORT mục 9 và WALKTHROUGH 5.7, kèm `scripts/induction_baselines.py` |
| R2 - hạ phát biểu cơ chế xuống mức độ | **Đã sửa.** Cả hai tài liệu giờ nêu 32,7-54,8% quãng đường, bỏ dạng "không phải X mà là Y" |
| R3 - nêu bất thường `nano` với `SYMBOL` | **Đã sửa.** Nêu kèm tính toán kỳ vọng dương tính giả (0,45 ô trên 9 phép kiểm) |

Ba mục đóng trong cùng lượt, 0 USD. Còn lại từ các vòng trước: R5 vòng 2 (đưa cảnh báo split `test-*` vào bản thảo) và phần chưa xong ở mục 11 REPORT.

### 11.8 Kết luận vòng 3

Luận điểm lõi **sống sót qua đợt tấn công mạnh nhất tôi dựng được**. Tôi đã thử quy hiệu ứng về ba thứ - số cạnh xuất ra, cấu trúc nhóm của phép hoán vị, và nhiễu thống kê - và không thứ nào giải thích được nó.

Cái không sống sót là **cách phát biểu**. Hai finding MAJOR đều là bệnh chung của bài phân tích: báo một thước đo mà không nói sàn của nó ở đâu, và biến một hiệu ứng có mức độ thành một mệnh đề nhị phân. Cả hai sửa trong một buổi, không tốn đồng nào.

Khoảng cách tới Accept giờ là ba lần viết lại và một bảng bổ sung.


---

## 12. Vòng 4 - kiểm dữ liệu đối chiếu bài CLadder gốc (2026-09-08)

Kích hoạt bởi yêu cầu của tác giả: đối chiếu bộ dữ liệu với arXiv:2312.04350.

### 12.1 Ba sự thật lấy từ bài gốc

| Kiểm chứng | Kết quả |
|---|---|
| Bài báo nêu v1.5 có **10.112** câu hỏi | `full_v1.5_default.csv` có **đúng 10.112** dòng. Đây là bản phát hành chính thức |
| Ba file `test-*` có 10.392 / 10.392 / 10.240 dòng | **Nhiều hơn cả bộ đầy đủ**, nên không thể là một split test của v1.5 |
| Cột của `full` so với `test-*` | `full` có 10 cột, `test-*` có 9 - thiếu `question_property`. Trình xem của HuggingFace hiện cũng lỗi vì "các file dữ liệu không cùng số cột" |

Chỉ **70/150** prompt của `test-commonsense` là tiền tố của một prompt trong `full`. Vậy các file `test-*` không phải bản cắt cụt của v1.5, chúng là một artefact khác, không có tài liệu, và không nằm trong bản phát hành mà bài báo mô tả. Kết luận vòng trước - chỉ dùng `full_v1.5_default.csv` - được xác nhận, và giờ có thêm căn cứ từ chính bài gốc.

### 12.2 Finding MAJOR mới

| # | Chiều | Mô tả | Evidence Anchor | Confidence |
|---|---|---|---|---|
| N3 | Hiệu lực cấu trúc của điều kiện đối chứng | **Điều kiện `KEEP` không phải "đặt tên hợp lẽ thường".** Cột `question_property` cho thấy `--drop-nonsense` giữ lại 3.129 dòng `anticommonsense` bên cạnh 3.141 dòng hợp lẽ thường. Trong mẫu 174 item thực dùng: **79 anticommonsense, 95 hợp lẽ, chỉ 9 là `commonsense` thuần.** Đường sàn của toàn bộ thang từ vựng vì thế đã bị bỏ prior đúng trên 45% số item, và mọi hiệu ứng đo được ở mục 4 tới 7 REPORT đều **bị pha loãng**. | `dataset: data/full_v1.5_default.csv cot question_property - anticommonsense=3129, mau 174 item co 79 anticommonsense` | 5 - kiểm trực tiếp trên cột metadata |

**Mức độ pha loãng, đo được:** `PERMUTE` so với `KEEP` ở `RAW`, gpt-4.1: **-7,52 pp** khi gộp chung so với **-14,13 pp** khi chỉ lấy nhóm có prior đúng. Hiệu ứng thật lớn gần gấp đôi con số đã báo cáo.

**Đây là finding có lợi cho tác giả.** Nó không lật kết luận nào, nó làm mọi kết luận mạnh lên. Nhưng nó vẫn là MAJOR vì con số đang in trong báo cáo sai về độ lớn, và vì một điều kiện đối chứng bị đặt tên sai suốt ba vòng phản biện mà không ghế nào phát hiện - kể cả tôi.

### 12.3 Hai điểm mạnh mới

**S1: Phân tầng biến confound thành một điều kiện thứ ba, và nó cho một phép kiểm có thể thất bại.**

| Điều kiện | Item có prior đúng | Item prior đã sai sẵn |
|---|---|---|
| `RAW` | **-12,23 pp, 8/9 đạt p<0,05** | -8,91 pp, 2/9 |
| `ORACLE` | **-3,84 pp, 0/9 đạt p<0,05** | -7,91 pp, 1/9 |

Cơ chế được nêu bắt buộc dự đoán rằng **xoá một prior đúng phải đắt hơn xoá một prior vốn đã sai**. Dự đoán đó có thể sai. Nó đã không sai.

Và luận điểm chính sắc hơn hẳn: trên nhóm item mà model **có** prior đúng để mất, cấp đồ thị đưa tác hại từ 8/9 ô có ý nghĩa xuống **0/9**. Bản gộp chung chỉ cho 8/9 xuống 3/9.
**Evidence Anchor**: `table: results/prior_strength.csv - ORACLE, prior dung, 0/9 dat p<0.05`

**S2: Một phép nhân bản độc lập, không hẹn mà có.**

Anticommonsense của CLadder là cùng một thao tác với `PERMUTE`: giữ từ thật, phá chiều nhân quả hợp lẽ. Khác nhóm tác giả, khác phương pháp, khác tập item.

| Model | Anticommonsense của CLadder | `PERMUTE` của dự án |
|---|---|---|
| gpt-4.1 | -11,6 pp | -14,1 pp |
| gpt-4.1-mini | -11,7 pp | -13,8 pp |
| gpt-4.1-nano | -3,6 pp | -4,6 pp |

Ba cặp, ba lần khớp, kể cả ở model lệch chuẩn là `nano`. `PERMUTE` không phải một thao tác tự chế cho ra hiệu ứng của riêng nó.
**Evidence Anchor**: `table: results/prior_strength.csv muc 1 - chenh prior dung so voi prior sai san co tren dieu kien KEEP/RAW`

### 12.4 Mục sửa bắt buộc

**R1: Phân tầng theo `question_property` ở mọi bảng từ vựng** - **đã sửa trong lượt.** Thêm `scripts/analyze_prior_strength.py`, mục 4.1 và 4.2 REPORT, khối tương ứng trong WALKTHROUGH 5.3, và `question_property` giờ được ghi vào output của `pilot.py`.

**R2: Ghi bẫy này vào sổ tay** - **đã sửa.** Bẫy 0c trong WALKTHROUGH phần 6: *cột metadata mà bạn chưa mở ra xem thường là cột quan trọng nhất*.

**R3: Nêu rõ nguồn gốc ba file `test-*`** - còn để mở. Nên hỏi nhóm CLadder qua HuggingFace discussion, vì trình xem của họ cũng đang lỗi vì chính vấn đề không đồng nhất số cột này. Đây là đóng góp cho cộng đồng, không chặn bài báo.

### 12.5 Kết luận vòng 4

Bốn vòng phản biện, và vòng này là vòng đầu tiên mà finding lớn nhất đến từ **yêu cầu của tác giả** chứ không phải từ hội đồng. Điều đó nên được ghi lại: ba vòng trước đều đọc cột `question_property` mà không ai mở nó ra.

Luận điểm không đổi, nhưng mọi con số của nó đều mạnh lên. Quyết định giữ **MINOR REVISION**.
