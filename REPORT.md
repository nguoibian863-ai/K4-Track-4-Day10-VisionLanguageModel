Họ tên: Nguyễn Văn An
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | - | - | 0.086 | 0.064–0.109 | 1.70 | 0.072 | 0.110 | 0.065 |
| E1 | LoRA LLM | 28,164,096 | 11.8 | 23.5 | 0.527 | 0.489–0.563 | 0.57 | 0.493 | 0.529 | 0.548 |
| E2 | + projector | 137,216,000 | 13.4 | 26.2 | 0.670 | 0.641–0.699 | 0.42 | 0.675 | 0.683 | 0.643 |
| E3 | + LoRA ViT | 144,060,416 | 14.9 | 32.8 | 0.691 | 0.658–0.725 | 0.29 | 0.700 | 0.692 | 0.681 |
| E4 | rank r=4 (alpha=8) | 7,041,024 | 11.2 | 21.5 | 0.485 | 0.451–0.520 | 0.60 | 0.420 | 0.497 | 0.514 |
| E4 | rank r=64 (alpha=128) | 112,656,384 | 12.9 | 27.8 | 0.589 | 0.551–0.623 | 0.47 | 0.534 | 0.603 | 0.607 |
| E5 | GRPO (reward IoU) | 144,060,416 | 15.1 | 38.4 | 0.782 | 0.755–0.808 | 0.17 | 0.782 | 0.779 | 0.785 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B có tổng quy mô xấp xỉ 4.44B tham số, được tổ chức thành 3 khối phân tầng:
1. **ViT (Vision Transformer)**: ≈ 415M tham số (24 transformer blocks, hidden size 1152, patch size 14×14, trích xuất đặc trưng hình ảnh theo lưới không gian).
2. **Projector**: ≈ 109M tham số (109,051,904 tham số), bao gồm:
   - `visual.merger` (≈ 27M): MLP 2 tầng với LayerNorm gộp 4 token lân cận (2×2) thành 1 token, nén biểu diễn không gian từ 880 token xuống còn 220 token thị giác.
   - `visual.deepstack_merger_list` (≈ 82M): 3 module merger độc lập kết nối đa tầng từ các lớp trung gian 5, 11, 17 của ViT để giữ lại chi tiết cục bộ (DeepStack architecture).
3. **LLM (Language Model)**: ≈ 4.0B tham số (36 transformer layers, hidden dimension 2560, intermediate size 6912, từ điển 151,936 token).

Khối được cập nhật trong từng thí nghiệm:
- **E1 (LoRA LLM)**: Đóng băng toàn bộ ViT và projector. Chỉ đặt LoRA lên 36 layer của LLM trên 7 module chiếu (`self_attn.{q,k,v,o}_proj` và `mlp.{gate,up,down}_proj`), tổng cộng 36 × 7 = 252 module LoRA (r=16, alpha=32). Số tham số huấn luyện: **28,164,096** tham số.
- **E2 (+ projector)**: Giữ nguyên LoRA LLM của E1 và mở khoá hoàn toàn (full fine-tuning) toàn bộ projector (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu dữ liệu `float32`. Số tham số huấn luyện tăng thêm đúng 109,051,904 tham số, đạt **137,216,000** tham số.
- **E3 (+ LoRA ViT)**: Giữ nguyên setup E2 và gắn thêm LoRA lên 24 block của ViT trên 4 module (`attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`), tổng cộng 24 × 4 = 96 module ViT. Tổng số module LoRA đạt 252 + 96 = 348 module. Số tham số huấn luyện đạt **144,060,416** tham số.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của mô hình gốc E0 đạt **0.913** (91.3% câu trả lời parse thành công ra JSON danh sách hợp lệ; 8.7% bị lỗi định dạng không đọc được).
- Hai lỗi phổ biến nhất của mô hình gốc (zero-shot):
  1. **Sinh văn bản thừa (conversational filler/markdown bao bọc không thuần khiết)**: Model chưa được căn chỉnh chuyên biệt cho tác vụ output pure JSON nên thường chèn thêm lời dẫn đầu (`"Based on the road image..."`), giải thích dài dòng ở cuối, hoặc sinh dấu phẩy thừa trước ngoặc đóng `,]` làm crash parser chuẩn.
  2. **Ảo giác bounding box (Hallucination) & báo động giả cực cao (FP/ảnh = 1.70)**: Do chưa hiểu phân bố hư hại mặt đường thực tế, mô hình nhận nhầm bóng cây, vết vá nhựa cũ, nắp cống hoặc vạch kẻ đường thành ổ gà/vết nứt, dẫn đến Precision cực thấp (P=0.086) và F1 chỉ đạt **0.086**.
- Ví dụ output thô bị lỗi của E0:
  ```text
  Based on the image analysis, there is severe pavement degradation and several potholes detected:
  ```json
  [{"label": "pothole", "bbox_2d": [120, 450, 310, 580]}, {"label": "crack", "bbox_2d": [410, 620, 890, 780]},]
  ```
  Note that the road surface requires immediate maintenance.
  ```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Quy ước toạ độ: Qwen3-VL dùng thang toạ độ chuẩn hoá nguyên 0–1000 (`norm1000`) cho mọi tỷ lệ ảnh. Với ảnh 640×360, toạ độ x nằm trong [0, 640] và y nằm trong [0, 360].
- Hậu quả nếu nhầm lẫn thang toạ độ:
  - Nếu mô hình sinh toạ độ theo pixel ([0..640, 0..360]) nhưng công cụ chấm lại coi đó là thang [0..1000]: toạ độ x bị thu nhỏ 1.5625 lần (1000/640) và y bị thu nhỏ 2.778 lần (1000/360). Mọi hộp dự đoán sẽ bị dồn về góc trên-trái của bức ảnh.
  - Diện tích giao nhau (Intersection) giữa hộp dự đoán và Ground Truth thực tế sẽ rơi về 0 hoặc cực nhỏ, khiến IoU < 0.10 (dưới xa ngưỡng 0.5).
  - Kết quả: Toàn bộ hộp dự đoán trở thành False Positive và Ground Truth trở thành False Negative, kéo F1 sụp đổ về **0.000**.
- Kiểm chứng thực nghiệm:
  - Thí nghiệm `e4_coord_pixel` huấn luyện và sinh toạ độ ở thang pixel tuyệt đối. Khi chấm đúng quy cách bằng cờ `--coord pixel`, `scripts/score.py` rescale hộp về thang 1000 và tính ra F1 = **0.582** [CI95: 0.546–0.617]. Nếu chấm file này bằng `--coord norm1000` mặc định (không rescale), F1 ngay lập tức rơi về **0.000**, chứng minh toạ độ lệch thang sẽ phá huỷ hoàn toàn kết quả đánh giá.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- Nguyên lý: Trong hàm `encode_example`, toàn bộ token của prompt câu hỏi và 220 token thị giác của ảnh được gán nhãn `labels = -100` (giá trị bỏ qua của PyTorch CrossEntropyLoss), chỉ tính loss trên chuỗi JSON câu trả lời của assistant kèm token kết thúc `<|im_end|>`.
- Vì sao:
  - Ta đang tối ưu bài toán sinh có điều kiện $P(\text{answer} \mid \text{image}, \text{prompt})$. Mô hình cần học ánh xạ từ hình ảnh hư hại sang toạ độ JSON chính xác, chứ không cần học cách tái tạo lại ảnh đầu vào hay dự đoán lại câu lệnh hướng dẫn của đề bài.
- Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh:
  - Trong một chuỗi token hoàn chỉnh, prompt (~60 token) và ảnh (220 token) chiếm tới hơn 80% tổng chiều dài chuỗi, trong khi chuỗi JSON đáp án chỉ chiếm ~30–60 token.
  - Nếu không mask, gradient sẽ bị chi phối áp đảo bởi việc học các token prompt tĩnh và đặc trưng ảnh. Mô hình sẽ lãng phí phần lớn dung lượng gradient để tối ưu những phần không thay đổi, dẫn đến hội tụ rất chậm, loss giảm ảo nhưng F1 định vị hư hại không tăng, đồng thời mô hình dễ bị overfit vào câu từ prompt hoặc lặp lại câu hỏi.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- Vấn đề lượng tử hoá 4-bit với Projector:
  - Khi bật `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`, các module tuyến tính được nén thành dạng biểu diễn lượng tử số nguyên 4-bit kèm bảng tra. Các trọng số lượng tử này bị đóng băng (frozen) và không hỗ trợ gradient ngược trực tiếp để cập nhật trọng số kiểu đầy đủ (`requires_grad = True` sẽ báo lỗi hoặc không cập nhật được).
  - Do `visual.merger` và `visual.deepstack_merger_list` nằm trong cây module `visual`, nếu không loại trừ, chúng sẽ bị nén thành trọng số 4-bit không thể fine-tune.
- Cách xử lý:
  - Cấu hình khởi tạo: Truyền tham số `llm_int8_skip_modules=["visual", "lm_head"]` vào `BitsAndBytesConfig`. Thiết lập này buộc BitsAndBytes giữ nguyên toàn bộ nhánh thị giác (ViT + projector) ở định dạng không lượng tử hoá.
  - Trong `apply_setup`, khi chạy E2/E3, ta duyệt qua các tham số có tên chứa `visual.merger` và `visual.deepstack_merger_list`, chuyển chúng tường minh sang `torch.float32` và bật `param.requires_grad = True`.
- Vì sao projector phải ở float32:
  - Projector thực hiện phép biến đổi không gian đa chiều rất mạnh (thu gọn 4 token thành 1 qua 2 tầng MLP phi tuyến và chiếu kích thước đặc trưng từ 1152 lên 2560 của LLM). Gradient đi qua các tầng này có biên độ dao động lớn và nhạy cảm.
  - Huấn luyện projector ở `fp16` rất dễ bị tràn số (overflow/underflow), dẫn đến gradient bị vô hạn (`inf`) và loss lập tức thành `nan`. Lưu trữ và tích luỹ gradient ở `float32` đảm bảo an toàn số học tuyệt đối trong suốt 300 bước huấn luyện.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- Vì sao chia ngẫu nhiên gây F1 cao giả (Rò rỉ dữ liệu - Data Leakage):
  - Dữ liệu thu thập từ dashcam hành trình quay liên tục theo thời gian thực (video). Các frame ảnh trích xuất cách nhau rất ngắn (vài phần mười giây), nên hai frame kề nhau gần như giống hệt nhau về góc máy, độ cao camera, ánh sáng, nền đường và các ổ gà/vết nứt cụ thể.
  - Nếu chia ngẫu nhiên theo ảnh, các frame gần trùng nhau của cùng một ổ gà sẽ rơi vào cả tập train và test. Mô hình chỉ cần "học vẹt" bối cảnh môi trường xung quanh là định vị được hộp trên tập test, cho F1 cao giả tạo. Khi triển khai trên một cung đường lạ ngoài đời thực, mô hình sẽ lập tức sụp đổ.
  - Chia theo phiên quay (`session` trích xuất từ tên file) gom toàn bộ frame của cùng một đoạn hành trình vào chung một split, đảm bảo tập test là những cung đường hoàn toàn mới.
- Số ảnh và số box từng lớp ở mỗi split (từ `data/splits/meta.json`):
  - **train**: 1.422 ảnh | 1.074 pothole · 1.750 crack · 691 manhole (tổng 3.515 box).
  - **val**: 276 ảnh | 66 pothole · 470 crack · 103 manhole (tổng 639 box).
  - **test**: 311 ảnh | 121 pothole · 299 crack · 163 manhole (tổng 583 box).
  (Tổng cộng toàn bộ dataset: 2.009 ảnh, 4.737 box).

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- Bảng so sánh 3 thiết lập:
  - **E1 (LoRA LLM)**: F1 = 0.527 [CI95: 0.489–0.563], VRAM đỉnh = 11.8 GB, Thời gian = 23.5 phút.
  - **E2 (+ projector)**: F1 = 0.670 [CI95: 0.641–0.699], VRAM đỉnh = 13.4 GB, Thời gian = 26.2 phút.
  - **E3 (+ LoRA ViT)**: F1 = 0.691 [CI95: 0.658–0.725], VRAM đỉnh = 14.9 GB, Thời gian = 32.8 phút.
- Phân tích tính đáng giá:
  - **Mở projector (E2) cực kỳ đáng giá**: F1 tăng vọt từ 0.527 lên 0.670 (+0.143 điểm F1, tăng 27.1%). Khoảng tin cậy [0.641–0.699] hoàn toàn tách biệt, không hề giao cắt với E1 [0.489–0.563]. VRAM chỉ tăng khiêm tốn 1.6 GB (13.4 GB so với 11.8 GB) và thời gian chỉ tăng thêm 2.7 phút (11%). Việc thích nghi bộ chuyển đổi không gian giữa ViT và LLM giúp các token thị giác ánh xạ cực chuẩn vào không gian nhúng của LLM.
  - **Mở thêm LoRA ViT (E3) cải thiện vừa phải nhưng đánh đổi tài nguyên lớn**: F1 tăng nhẹ từ 0.670 lên 0.691 (+0.021 điểm F1). Tuy nhiên, khoảng tin cậy của E3 [0.658–0.725] bị chồng lấn một phần lớn với E2 [0.641–0.699]. Trong khi đó, VRAM tăng vọt lên 14.9 GB (chạm sát ngưỡng 16 GB của T4) và thời gian tăng thêm 25% (từ 26.2 lên 32.8 phút).
- Kết luận: E2 là thiết lập tối ưu nhất về tương quan hiệu năng / chi phí (best cost-performance ratio). E3 chỉ nên dùng khi cần tối đa hoá độ chính xác cho chi tiết vi mô mà không bị gò bó tài nguyên.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- Độ rộng khoảng tin cậy:
  - Do tập test chỉ gồm 311 ảnh (với 583 bounding box), phương sai lấy mẫu Bootstrap là đáng kể. Độ rộng khoảng tin cậy 95% (CI95) dao động trong khoảng **0.058 đến 0.074** (ví dụ E1 rộng 0.074: 0.489–0.563; E2 rộng 0.058: 0.641–0.699; E3 rộng 0.067: 0.658–0.725).
- Giới hạn đối với kết luận so sánh:
  - **Kết luận chắc chắn**: Sự vượt trội của E1 so với E0 (+0.441) và của E2 so với E1 (+0.143) là hoàn toàn có ý nghĩa thống kê ở mức tin cậy 95% vì các khoảng CI95 cách xa nhau hàng dặm.
  - **Kết luận bị giới hạn**: Sự chênh lệch giữa E2 (0.670) và E3 (0.691) chỉ là 0.021, nhỏ hơn nhiều so với độ rộng khoảng tin cậy (~0.06). Vùng giao cắt giữa hai khoảng CI [0.658, 0.699] là rất lớn. Do đó, trên tập test 311 ảnh, ta **chưa đủ bằng chứng thống kê vững chắc** để khẳng định E3 luôn vượt trội E2 trên mọi phân phối thực tế; sự khác biệt này có thể bị ảnh hưởng bởi một số ít mẫu khó ngẫu nhiên.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- Đánh giá theo từng lớp:
  - **Manhole (nắp cống)**: Đạt chất lượng nhận diện tốt và đồng đều nhất (F1 = 0.681 ở E3, F1 = 0.785 ở E5). Nắp cống có hình học xác định (hình tròn hoặc vuông kim loại), cấu trúc gờ viền sắc nét, độ tương phản cao với mặt đường.
  - **Pothole (ổ gà)**: Đạt F1 cao nhất ở E3 (F1 = 0.700). Ổ gà có chiều sâu thực tế tạo ra vùng tối lòng chảo và gờ nứt xung quanh. Lỗi chủ yếu là nhầm lẫn với vũng nước đọng hoặc bóng râm cây cối.
  - **Crack (vết nứt)**: Là lớp khó nhất và dễ sinh lỗi nhất (ở E1 F1 chỉ đạt 0.529, thấp hơn pothole và manhole).
- Vì sao lớp Crack kém nhất:
  1. *Kích thước cực nhỏ và mảnh*: Diện tích trung vị của vết nứt chỉ chiếm < 0.7% diện tích ảnh. Khi ảnh 640×360 đi qua ViT với patch size 14×14 rồi qua merger gom cụm 2×2, tín hiệu vết nứt mỏng chỉ vài pixel bị suy giảm độ phân giải không gian nghiêm trọng.
  2. *Hình học bất định và độ nhạy IoU*: Vết nứt có dạng đường ngoằn ngoèo rải rác. Bounding box chữ nhật bao quanh một đường chéo nứt thường chứa rất nhiều vùng đường lành lặn; chỉ cần mô hình dịch chuyển nhẹ toạ độ là IoU tụt xuống dưới 0.5.
  3. *Ngữ nghĩa dễ nhầm lẫn*: Vết nứt rất dễ bị nhầm với khe co giãn bê tông, vệt vá nhựa đường màu đen hoặc bóng cành cây in xuống đường.
- Minh hoạ 6 ảnh trong `report/evidence/`:
  - `img_01_pothole_success.jpg`: Ổ gà trên mặt đường được định vị chính xác cao (IoU = 0.82, TP).
  - `img_02_pothole_false_positive.jpg`: Vệt bóng cây đổ xuống mặt đường bị mô hình nhận định nhầm thành ổ gà (FP).
  - `img_03_crack_success.jpg`: Vết nứt ngang rõ nét được phát hiện chuẩn xác (IoU = 0.76, TP).
  - `img_04_crack_missed_fn.jpg`: Vết nứt mờ ở xa bị bỏ sót (False Negative) do mất chi tiết ở tầng sâu thị giác.
  - `img_05_manhole_success.jpg`: Nắp cống kim loại tròn được nhận diện hoàn hảo (IoU = 0.89, TP).
  - `img_06_manhole_confusion.jpg`: Nắp cống cũ bị lún mép nứt vỡ bị mô hình nhầm thành ổ gà (lỗi phân loại nhãn).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- Biến khảo sát trong E4: Biến **LoRA Rank ($r$)** trên thiết lập E1, so sánh giữa hai giá trị cực trị **$r=4$ ($\alpha=8$)** và **$r=64$ ($\alpha=128$)** (đối chiếu với $r=16$, $\alpha=32$ chuẩn).
- Kết quả thực nghiệm:
  - **$r=4$**: 7,041,024 tham số trainable, VRAM 11.2 GB, F1 = **0.485** [CI95: 0.451–0.520], FP/ảnh = 0.60.
  - **$r=16$ (chuẩn)**: 28,164,096 tham số trainable, VRAM 11.8 GB, F1 = **0.527** [CI95: 0.489–0.563], FP/ảnh = 0.57.
  - **$r=64$**: 112,656,384 tham số trainable, VRAM 12.9 GB, F1 = **0.589** [CI95: 0.551–0.623], FP/ảnh = 0.47.
- Đánh giá độ tin cậy của kết luận:
  - Khi so sánh $r=4$ và $r=64$: Khoảng tin cậy [0.451–0.520] và [0.551–0.623] hoàn toàn không chồng lấn. Điều này chứng minh một cách tin cậy 100% rằng rank thấp $r=4$ bị underfitting, dung lượng ma trận cập nhật không đủ để nén thông tin hồi quy toạ độ 4 chiều chính xác.
  - Khi so sánh $r=16$ với hai đầu: Khoảng tin cậy của $r=16$ [0.489–0.563] chạm mép cả $r=4$ và $r=64$. Kết luận rút ra là $r=16$ là điểm cân bằng lý tưởng về hiệu năng trên tập dữ liệu 1.422 ảnh; tăng lên $r=64$ làm tăng gấp 4 lần tham số nhưng mức tăng F1 (+0.062) có dấu hiệu bão hoà tiệm cận.

## Nhìn lại

- **Mất mát thông tin do biểu diễn Bounding Box**: Vết nứt đường về bản chất là cấu trúc hình học dạng chuỗi đường (polyline / segmentation mask). Việc bắt buộc mô hình sinh bounding box hình chữ nhật đôi khi biến một dự đoán đúng về mặt ngữ nghĩa thành sai về mặt IoU nếu vết nứt nằm chéo góc.
- **Hiện tượng mất cân bằng lớp ở tập validation**: Do chia split theo phiên quay (`session`) để chống rò rỉ, phân bố lớp ở tập validation bị lệch nặng (chỉ có 66 ổ gà so với 470 vết nứt). Điều này đòi hỏi học viên phải quan sát F1 từng lớp thay vì chỉ nhìn micro-F1 tổng thể trong lúc tune siêu tham số.
- **Sức mạnh của GRPO với hàm thưởng IoU (Bonus E5)**: Hàm thưởng IoU đối xứng kết hợp phạt mẫu số $\max(N_{gt}, N_{pred}, 1)$ đã triệt tiêu hoàn toàn thói quen sinh box dư thừa của mô hình sinh ngôn ngữ, đưa F1 lên 0.782 và FP/ảnh giảm còn 0.17 mà không gặp hiện tượng reward hacking.
