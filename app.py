import gradio as gr
import cv2
import numpy as np
from utils.preprocess import preprocess_leaf_image
from utils.feature_extraction import extract_leaf_features
from utils.search import LeafSearch
from utils.database import LeafDatabaseSQL 

def query_leaf(image):
    # Tiền xử lý và trích đặc trưng
    processed = preprocess_leaf_image(image)
    feature = extract_leaf_features(processed)

    # Truy xuất ảnh tương đồng
    db = LeafDatabaseSQL()
    db.load_all_features(is_query=False)
    retrieval = LeafSearch(db)
    results = retrieval.search_with_feature_vector(feature, top_n=3)
    db.close()

    outputs = []
    for r in results:
        img = cv2.imread(r['path'])[..., ::-1]  # BGR to RGB
        similarity = round(r['similarity'] * 100, 2)
        distance = round(1 - r['similarity'], 4)
        desc = f"Tương đồng: {similarity}%\nKhoảng cách: {distance}"
        outputs.append(img)
        outputs.append(desc)

    return tuple(outputs)

gr.Interface(
    fn=query_leaf,
    inputs=gr.Image(type="filepath", label="Tải ảnh lá lên"),
    outputs=[
        gr.Image(label="Kết quả 1"), gr.Textbox(label="Thông tin 1"),
        gr.Image(label="Kết quả 2"), gr.Textbox(label="Thông tin 2"),
        gr.Image(label="Kết quả 3"), gr.Textbox(label="Thông tin 3")
    ],
    title="Leaf Image Retrieval",
    description="Tải ảnh lá để tìm 3 ảnh gần giống nhất trong cơ sở dữ liệu."
).launch()