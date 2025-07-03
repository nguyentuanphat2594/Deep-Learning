# Deep-Learning
## 1. CNN - MNIST Digit Classification
Mục tiêu: Phân loại chữ số viết tay từ tập dữ liệu MNIST.
Thuật toán: Convolutional Neural Network (CNN)
Xử lý ảnh:
Chuẩn hóa pixel
Reshape ảnh về (28,28,1)
One-hot encode nhãn đầu ra

Mô hình:
2 lớp Conv2D + MaxPooling2D
Flatten → Dense → Softmax
Kết quả: Accuracy cao trên cả tập validation và test.
Trực quan hóa: Độ chính xác & mất mát theo epoch, ảnh dự đoán so với nhãn thật.

## 2. RNN / LSTM / GRU - Clothing Review Recommendation
Mục tiêu: Dự đoán người dùng có khuyến nghị sản phẩm hay không dựa trên review text.
Xử lý dữ liệu:
Tiền xử lý Review Text, Title (xóa stopwords, lemmatize, loại bỏ dấu câu)
Kết hợp review, tiêu đề và tên lớp sản phẩm thành cột 'Text'
Vector hóa bằng Tokenizer + Padding

Mô hình:
So sánh 3 mạng: RNN, LSTM và GRU
Sử dụng Embedding + Dropout + Dense
Đánh giá:
Biểu đồ độ chính xác train/val qua epochs cho từng mô hình
So sánh trực quan hiệu quả RNN vs LSTM vs GRU
Kết luận: GRU thường học nhanh và hiệu quả tốt trong tập này.

## 3. ANN - Google Analytics Revenue Prediction
Mục tiêu: Dự đoán doanh thu từ dữ liệu người dùng Google Analytics.
Xử lý dữ liệu:
Phân tách dữ liệu JSON từ các cột như device, geoNetwork, totals
Tạo các đặc trưng nhóm theo fullVisitorId: mean_hits_per_day, sum_pageviews_per_day, v.v.
Tách train/val theo ngày, xử lý missing và chuẩn hóa với MinMaxScaler

Mô hình: Artificial Neural Network
2 lớp Dense ẩn (ReLU), 1 lớp đầu ra dự đoán doanh thu
Dùng Adam optimizer, hàm loss: MSE
Kết quả:
Dự đoán trên tập validation và test
Lưu output dự đoán dạng log doanh thu (log1p)
Trực quan hóa: Revenue theo năm/tháng/ngày/thứ, biểu đồ đánh giá mô hình.
