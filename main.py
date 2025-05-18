import os
import cv2
import argparse
from utils.search import LeafSearch
from utils.database import LeafDatabaseSQL 
from utils.visualization import create_result_visualization


def main():
    parser = argparse.ArgumentParser(description="Leaf Image Search System with MySQL")
    parser.add_argument("--query", type=str, help="Đường dẫn ảnh truy vấn")
    parser.add_argument("--top_k", type=int, default=3, help="Số ảnh kết quả tương tự cần trả về")
    parser.add_argument("--build_db", action="store_true", help="Tạo database MySQL từ thư mục 'dataset/'")
    parser.add_argument("--no_show", action="store_true", help="Không hiển thị ảnh kết quả lên màn hình")
    parser.add_argument("--save_result", action="store_true", help="Lưu ảnh kết quả vào thư mục 'results'")
    args = parser.parse_args()

    db = LeafDatabaseSQL()
    db.create_tables()

    if args.build_db:
        print(">>> Đang xử lý và lưu đặc trưng dataset vào MySQL...")
        db.process_directory("dataset")
        print(">>> Đã hoàn tất tạo database.")
        db.close()
        return

    if not args.query:
        print(">>> Vui lòng cung cấp ảnh truy vấn với --query <path_to_image>")
        return

    # Xử lý ảnh truy vấn và lưu vào bảng query
    db.process_and_insert_image(args.query, label="query", is_query=True)

    # Load dataset
    db_dataset = LeafDatabaseSQL()
    db_dataset.load_all_features(is_query=False)

    # Load query
    db_query = LeafDatabaseSQL()
    db_query.load_all_features(is_query=True)
    if len(db_query.features) == 0:
        print(">>> Không có đặc trưng nào trong bảng truy vấn.")
        return
    query_feature = db_query.features[-1]  # Lấy ảnh truy vấn mới nhất

    # Truy xuất
    search = LeafSearch(db_dataset)
    results = search.search_with_feature_vector(query_feature, top_n=args.top_k)

    print("\n>>> KẾT QUẢ TRUY XUẤT:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r['label']} ({r['path']}) - Similarity: {r['similarity']:.2f}")

    if args.save_result or not args.no_show:
        os.makedirs("results", exist_ok=True)
        result_img_path = os.path.join("results", f"result_{os.path.splitext(os.path.basename(args.query))[0]}.jpg")
        create_result_visualization(args.query, results, output_path=result_img_path if args.save_result else None, show=not args.no_show)

    db.close()


if __name__ == "__main__":
    main()