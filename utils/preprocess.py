import numpy as np
import cv2
import os

def preprocess_leaf_image(image_path, output_path=None):
    """
    Tiền xử lý ảnh lá: tách lá, giữ viền, nền trắng.
    Cải tiến:
    - Mở rộng khoảng HSV để bao phủ nhiều sắc thái màu xanh (cả sáng và tối)
    - Thêm xử lý ngoại lệ khi không tìm thấy lá
    - Morphology tối ưu với các kernel phù hợp
    - Làm sắc nét viền lá chỉ vùng biên, dilation 2 lần để viền rõ hơn
    - Lưu ảnh nếu có output_path
    """
    # Đọc ảnh và chuyển sang RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển sang HSV bằng OpenCV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Khai báo dải màu HSV cho nhiều sắc thái xanh
    lower_green1 = np.array([30, 50, 50])
    upper_green1 = np.array([90, 255, 255])
    lower_green2 = np.array([30, 30, 30])
    upper_green2 = np.array([90, 150, 150])

    # Tạo mask cho 2 dải màu rồi kết hợp
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    combined_mask = cv2.bitwise_or(mask1, mask2)

    # Morphological close và dilate để làm sạch và mở rộng mask
    kernel_close = np.ones((5,5), np.uint8)
    kernel_dilate = np.ones((3,3), np.uint8)
    mask_clean = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    mask_clean = cv2.dilate(mask_clean, kernel_dilate, iterations=1)

    # Tìm contours
    contours, _ = cv2.findContours(mask_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Không tìm thấy lá trong ảnh")

    largest = max(contours, key=cv2.contourArea)

    # Tạo mask từ contour lớn nhất
    refined_mask = np.zeros_like(mask_clean)
    cv2.drawContours(refined_mask, [largest], -1, 255, thickness=-1)

    # Flood fill từ trọng tâm contour
    M = cv2.moments(largest)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        h, w = refined_mask.shape
        flood_fill_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(refined_mask, flood_fill_mask, (cx, cy), 255)

    # Tạo nền trắng
    white_bg = np.ones_like(image, dtype=np.uint8) * 255
    result = np.where(refined_mask[..., None] == 255, image, white_bg)

    # Tạo edge mask từ contour
    edge_mask = np.zeros_like(refined_mask)
    cv2.drawContours(edge_mask, [largest], -1, 255, 1)
    edge_mask = cv2.dilate(edge_mask, np.ones((3,3), np.uint8), iterations=2)

    # Làm sắc nét viền (sharpen) chỉ ở vùng viền
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(result, -1, sharpen_kernel)
    final = np.where(edge_mask[..., None] > 0, sharpened, result)

    # Lưu ảnh nếu có đường dẫn
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    return final
