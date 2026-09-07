CAC FILE CO TIEN TO "INVALID_no_question_" KHONG DUNG DUOC.

Chung duoc sinh ra bang cach chay tren test-noncommonsense-v1.5.csv, ma file do
KHONG CHUA CAU HOI: 0.00% prompt cua no co dau '?', trong khi
full_v1.5_default.csv la 100%.

Model duoc yeu cau tra loi yes/no cho mot prompt khong hoi gi ca, nen no chi co
the doan. Ket qua ~50% khong phai mot phat hien ve nang luc suy luan nhan qua
cua LLM; no la he qua toan hoc cua viec buoc doan.

Cot 'correct' trong cac file nay vo nghia. Cot f1 / precision / recall cua
induction it bi anh huong hon (viec dung do thi chi can phan boi canh, ma phan
do co trong file), nhung mau lai lay tu mot pool item khac han nen van khong so
sanh duoc voi nhanh chinh.

Thay the: thi nghiem tu vung ghep cap, doi ten bien NGAY TRONG cung mot item.
  src/lexical.py, scripts/analyze_lexical.py
  results/pilot_raw_lexKEEP.csv, pilot_raw_lexSYMBOL.csv, pilot_raw_lexPSEUDO.csv

Giu lai de doi chieu, khong dung de bao cao. Xem REPORT.md muc 2.
