import numpy as np

class LeafSearch:
    def __init__(self, database):
        """
        Hệ thống truy xuất ảnh lá sử dụng Cosine Similarity.
        
        Args:
            database: đối tượng có .features (list ndarray), .image_paths, .labels
        """
        self.database = database
        self.features = np.array(database.features)

        # Tính mean và std để chuẩn hóa
        self.mean = np.mean(self.features, axis=0)
        self.std = np.std(self.features, axis=0) + 1e-8  # tránh chia cho 0

        # Chuẩn hóa database
        self.norm_features = (self.features - self.mean) / self.std

    def normalize_vector(self, vector):
        """Chuẩn hóa vector truy vấn bằng mean/std đã học"""
        return (np.array(vector) - self.mean) / self.std

    def compute_cosine_distance(self, vec1, vec2):
        """Tính khoảng cách cosine thủ công"""
        dot = np.dot(vec1, vec2)
        norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return 1 - dot / (norm + 1e-8)

    def search_with_feature_vector(self, query_features, top_n=3):
        """
        Truy xuất top-N ảnh giống nhất từ ảnh truy vấn.

        Args:
            query_features: vector đặc trưng đã flatten (ndarray hoặc list)
            top_n: số kết quả trả về

        Returns:
            List dict {path, label, similarity}
        """
        query_norm = self.normalize_vector(query_features)
        distances = []

        for i, db_vec in enumerate(self.norm_features):
            dist = self.compute_cosine_distance(query_norm, db_vec)
            distances.append((i, dist))

        # Sắp xếp tăng dần theo khoảng cách
        distances.sort(key=lambda x: x[1])

        results = [{
            'path': self.database.image_paths[i],
            'label': self.database.labels[i],
            'similarity': 1.0 - d
        } for i, d in distances[:top_n]]

        return results
