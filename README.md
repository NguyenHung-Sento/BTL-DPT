# Tìm kiếm Ảnh Lá Cây

## Cài đặt

```bash
# Khởi tạo môi trường với venv 
python -m venv venv 

# Kích hoạt môi trường
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate  # Windows

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

## Cách sử dụng


### 1. Xây dựng cơ sở dữ liệu

Xây dựng cơ sở dữ liệu các đặc trưng từ ảnh đã tiền xử lý:

```bash
python main.py --build_db
```

### 2. Tìm kiếm ảnh tương tự

Tìm kiếm ảnh tương tự dựa trên một ảnh query:

```bash
python main.py --save_result --query test_images/Leaf-type-in-db/83.jpg

# Kết quả trả về lưu tại folder results
```

Tùy chọn:

- `--top_k`: Số ảnh kết quả trả về (mặc định: 3)
- `--no_show`: Không hiển thị ảnh kết quả lên màn hình

### 3. Chạy trên web

Upload ảnh và hiển thị tìm kiếm trên web Gradio:

```bash
python app.py
```