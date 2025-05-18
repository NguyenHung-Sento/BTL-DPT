import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_result_visualization(query_image_path, results, output_path=None, show=True):
    """
    Hiển thị trực quan ảnh truy vấn và 3 ảnh kết quả tương đồng nhất.
    - query_image_path: đường dẫn ảnh truy vấn
    - results: danh sách dict gồm 'path', 'label', 'similarity'
    - output_path: nơi lưu hình ảnh kết quả (nếu có)
    - show: có hiển thị hình ảnh hay không
    """
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 5))

    # Ảnh truy vấn bên trái
    plt.subplot(1, 4, 1)
    plt.imshow(query_image)
    plt.title("Ảnh truy vấn")
    plt.axis('off')

    # 3 ảnh kết quả
    for i, result in enumerate(results[:3]):
        result_image = cv2.imread(result['path'])
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 4, i + 2)
        plt.imshow(result_image)
        plt.title(f"Kết quả {i+1}\n{result['label']}\n{result['similarity']*100:.1f}%")
        plt.axis('off')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f">>> Đã lưu hình ảnh kết quả vào {output_path}")

    if show:
        plt.show()
    else:
        plt.close()