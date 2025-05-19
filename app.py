import gradio as gr
import cv2
import numpy as np
from utils.preprocess import preprocess_leaf_image
from utils.feature_extraction import extract_leaf_features
from utils.search import LeafSearch
from utils.database import LeafDatabaseSQL 

def query_leaf(image):
    # Truy xuất ảnh tương đồng
    db = LeafDatabaseSQL()
    db.process_and_insert_image(image, label="query", is_query=True)

    # Load dataset
    db_dataset = LeafDatabaseSQL()
    db_dataset.load_all_features(is_query=False)

    # Load query
    db_query = LeafDatabaseSQL()
    db_query.load_all_features(is_query=True)

    query_feature = db_query.features[-1]

    # Truy xuất
    search = LeafSearch(db_dataset)
    results = search.search_with_feature_vector(query_feature, top_n=3)
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