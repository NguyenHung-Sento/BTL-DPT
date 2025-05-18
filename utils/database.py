import pymysql
import numpy as np
import json
import os
import pandas as pd
from utils.preprocess import preprocess_leaf_image
from utils.feature_extraction import extract_leaf_features

class LeafDatabaseSQL:
    def __init__(self, host='localhost', user='root', password='123456', database='leafdb'):
        self.conn = pymysql.connect(host=host, user=user, password=password, database=database)
        self.cursor = self.conn.cursor()
        self.features = []
        self.image_paths = []
        self.labels = []

        self.feature_names = [
            # Color (HSV histograms + mean + std)
            *[f"h_bin_{i}" for i in range(8)],
            *[f"s_bin_{i}" for i in range(8)],
            *[f"v_bin_{i}" for i in range(8)],
            "h_mean", "s_mean", "v_mean",
            "h_std", "s_std", "v_std",

            # Shape (7 Hu + ratio + aspect ratio)
            *[f"hu_{i}" for i in range(7)], "perim_area_ratio", "aspect_ratio",

            # Texture (LBP histogram)
            *[f"lbp_bin_{i}" for i in range(10)],

            # Vein (density, consistency, mean_width, var_width)
            "vein_density", "vein_direction_consistency", "vein_mean_width", "vein_var_width",

            # Edge (roughness, angle_variance, edge_complexity)
            "edge_roughness", "edge_angle_variance", "edge_complexity"
        ]

    def create_tables(self):
        sql_dataset = '''
        CREATE TABLE IF NOT EXISTS leaf_features_dataset (
            id INT AUTO_INCREMENT PRIMARY KEY,
            path TEXT,
            label VARCHAR(255),
            feature_vector TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''
        sql_query = '''
        CREATE TABLE IF NOT EXISTS leaf_features_query (
            id INT AUTO_INCREMENT PRIMARY KEY,
            path TEXT,
            label VARCHAR(255),
            feature_vector TEXT
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        '''
        self.cursor.execute(sql_dataset)
        self.cursor.execute(sql_query)
        self.conn.commit()

    def clear_dataset_table(self):
        self.cursor.execute("DELETE FROM leaf_features_dataset")
        self.conn.commit()
        csv_path = "database/feature_database.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def clear_query_table(self):
        self.cursor.execute("DELETE FROM leaf_features_query")
        self.conn.commit()
        csv_path = "database/query_features.csv"
        if os.path.exists(csv_path):
            os.remove(csv_path)

    def insert_feature(self, path, label, feature_vector, is_query=False):
        table = "leaf_features_query" if is_query else "leaf_features_dataset"
        feature_str = json.dumps(feature_vector.tolist())
        sql = f"INSERT INTO {table} (path, label, feature_vector) VALUES (%s, %s, %s)"
        self.cursor.execute(sql, (path, label, feature_str))
        self.conn.commit()

        # Ghi thêm vào file CSV để tiện quan sát với tên cột
        csv_path = "database/query_features.csv" if is_query else "database/feature_database.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        row = [path, label] + feature_vector.tolist()
        columns = ["path", "label"] + self.feature_names
        df = pd.DataFrame([row], columns=columns)
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=header)

    def process_and_insert_image(self, image_path, label, is_query=False):
        try:
            if is_query:
                processed_img = preprocess_leaf_image(image_path,"results/query_processed.jpg")
            else:
                processed_img = preprocess_leaf_image(image_path)
            feature_vector = extract_leaf_features(processed_img)
            self.insert_feature(image_path, label, feature_vector, is_query=is_query)
            print(f">>> Đã xử lý và lưu đặc trưng cho ảnh: {image_path}")
        except Exception as e:
            print(f">>> Lỗi xử lý ảnh {image_path}: {e}")

    def process_directory(self, input_dir):
        # Xoá dữ liệu cũ trước khi xử lý lại
        self.clear_dataset_table()
        self.clear_query_table()
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, file)
                    label = os.path.basename(os.path.dirname(path))
                    self.process_and_insert_image(path, label, is_query=False)

    def load_all_features(self, is_query=False):
        table = "leaf_features_query" if is_query else "leaf_features_dataset"
        sql = f"SELECT path, label, feature_vector FROM {table}"
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        self.image_paths = []
        self.labels = []
        self.features = []
        for path, label, feature_str in rows:
            self.image_paths.append(path)
            self.labels.append(label)
            self.features.append(np.array(json.loads(feature_str), dtype=np.float32))
        self.features = np.array(self.features)

    def close(self):
        self.cursor.close()
        self.conn.close()
