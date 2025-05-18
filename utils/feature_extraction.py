import numpy as np
import cv2

# ==== MÀU SẮC ==== (8 bins HSV + mean + std = 30)
def rgb_to_hsv_manual(rgb):
    """
    Chuyển đổi ảnh từ RGB sang HSV.
    Trả về ảnh HSV với kênh H trong [0,360], S,V trong [0,1]
    """
    rgb = rgb.astype('float32') / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    # Tính hue theo công thức chuẩn HSV tùy theo max channel là R, G hay B
    h = np.zeros_like(cmax)
    mask = delta != 0
    idx = (cmax == r) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / delta[idx]) + 360) % 360
    idx = (cmax == g) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / delta[idx]) + 120) % 360
    idx = (cmax == b) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / delta[idx]) + 240) % 360

    s = np.zeros_like(cmax)
    s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]   # tính saturation
    v = cmax                                            # value = max channel
    return np.stack([h, s, v], axis=-1)

def histogram_manual(channel, bins, range_):
    """
    Tính histogram cho 1 kênh ảnh.
    channel: mảng giá trị pixel
    bins: số lượng bins histogram
    range_: tuple (min, max) giới hạn giá trị kênh
    """
    hist = np.zeros(bins)
    step = (range_[1] - range_[0]) / bins

    # Tính bin index cho mỗi pixel
    idxs = ((channel.flatten() - range_[0]) / step).astype(int)
    idxs = np.clip(idxs, 0, bins - 1)

    # Đếm số pixel rơi vào mỗi bin
    for idx in idxs:
        hist[idx] += 1
    
    # Chuẩn hóa histogram (tổng bằng 1)
    return hist / (np.sum(hist) + 1e-7)

def extract_color_features(image):
    """
    Trích xuất đặc trưng màu từ ảnh RGB.
    Bao gồm histogram H/S/V (mỗi 8 bins), trung bình và độ lệch chuẩn từng kênh HSV.
    """
    hsv = rgb_to_hsv_manual(image)
    h_hist = histogram_manual(hsv[..., 0], 8, (0, 360))
    s_hist = histogram_manual(hsv[..., 1], 8, (0, 1))
    v_hist = histogram_manual(hsv[..., 2], 8, (0, 1))
    mean = np.mean(hsv.reshape(-1, 3), axis=0)
    std = np.std(hsv.reshape(-1, 3), axis=0)
    return np.concatenate([h_hist, s_hist, v_hist, mean, std])

# ==== HÌNH DẠNG ==== (7 Hu moments + aspect_ratio + perimeter/area = 9)
def compute_raw_moments(cnt):
    """
    Tính các moment nguyên thủy (raw moments) của contour.
    cnt: danh sách điểm contour [(x,y), ...]
    Trả về dict các moment m00, m10, m01, ...
    """
    M = {'m00': 0, 'm10': 0, 'm01': 0, 'm11': 0, 'm20': 0, 'm02': 0,
         'm30': 0, 'm03': 0, 'm12': 0, 'm21': 0}
    for x, y in cnt:
        M['m00'] += 1
        M['m10'] += x
        M['m01'] += y
        M['m11'] += x * y
        M['m20'] += x ** 2
        M['m02'] += y ** 2
        M['m30'] += x ** 3
        M['m03'] += y ** 3
        M['m12'] += x * y ** 2
        M['m21'] += x ** 2 * y
    return M

def compute_central_moments(M):
    """
    Tính moment trung tâm từ moment nguyên thủy.
    """
    cx = M['m10'] / M['m00']    # tâm x
    cy = M['m01'] / M['m00']    # tâm y
    mu = {
        'mu20': M['m20'] - cx * M['m10'],
        'mu02': M['m02'] - cy * M['m01'],
        'mu11': M['m11'] - cx * M['m01'],
        'mu30': M['m30'] - 3 * cx * M['m20'] + 2 * cx ** 2 * M['m10'],
        'mu03': M['m03'] - 3 * cy * M['m02'] + 2 * cy ** 2 * M['m01'],
        'mu21': M['m21'] - 2 * cx * M['m11'] - cy * M['m20'] + 2 * cx ** 2 * M['m01'],
        'mu12': M['m12'] - 2 * cy * M['m11'] - cx * M['m02'] + 2 * cy ** 2 * M['m10']
    }
    return mu

def compute_hu_manual(mu):
    """
    Tính 7 Hu moments từ moment trung tâm.
    """
    eps = 1e-10
    hu = [
        mu['mu20'] + mu['mu02'],
        (mu['mu20'] - mu['mu02'])**2 + 4 * mu['mu11']**2,
        (mu['mu30'] - 3 * mu['mu12'])**2 + (3 * mu['mu21'] - mu['mu03'])**2,
        (mu['mu30'] + mu['mu12'])**2 + (mu['mu21'] + mu['mu03'])**2,
        (mu['mu30'] - 3 * mu['mu12']) * (mu['mu30'] + mu['mu12']) * ((mu['mu30'] + mu['mu12'])**2 - 3 * (mu['mu21'] + mu['mu03'])**2) +
        (3 * mu['mu21'] - mu['mu03']) * (mu['mu21'] + mu['mu03']) * (3 * (mu['mu30'] + mu['mu12'])**2 - (mu['mu21'] + mu['mu03'])**2),
        (mu['mu20'] - mu['mu02']) * ((mu['mu30'] + mu['mu12'])**2 - (mu['mu21'] + mu['mu03'])**2) +
        4 * mu['mu11'] * (mu['mu30'] + mu['mu12']) * (mu['mu21'] + mu['mu03']),
        (3 * mu['mu21'] - mu['mu03']) * (mu['mu30'] + mu['mu12']) * ((mu['mu30'] + mu['mu12'])**2 - 3 * (mu['mu21'] + mu['mu03'])**2) -
        (mu['mu30'] - 3 * mu['mu12']) * (mu['mu21'] + mu['mu03']) * (3 * (mu['mu30'] + mu['mu12'])**2 - (mu['mu21'] + mu['mu03'])**2)
    ]
    # Để tránh log của số âm hoặc 0, thêm eps, lấy -sign để chuẩn hóa
    return -np.sign(hu) * np.log10(np.abs(hu) + eps)

def extract_shape_features(image):
    """
    Trích xuất đặc trưng hình dạng từ ảnh RGB.
    - Chuyển ảnh về grayscale, tạo ảnh nhị phân
    - Tìm contour điểm non-zero
    - Tính 7 Hu moments + aspect ratio + perimeter/area
    Tổng 9 chiều
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # tạo mask: pixel tối (lá) là 1, nền là 0
    binary = (gray < 250).astype(np.uint8)

    # lấy tọa độ pixel biên (non-zero)
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return np.zeros(9)      # nếu không có biên trả về 0
    contour = list(zip(xs, ys)) # chuyển về danh sách điểm contour (x,y)
    M = compute_raw_moments(contour)
    if M['m00'] == 0:
        return np.zeros(9)
    mu = compute_central_moments(M)
    hu = compute_hu_manual(mu)

    # tính aspect ratio của bounding box
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    w, h = x_max - x_min + 1, y_max - y_min + 1
    aspect_ratio = w / (h + 1e-5)

    # diện tích xấp xỉ = số điểm contour
    area = len(contour)

    # tính perimeter (chu vi) gần đúng = tổng khoảng cách giữa các điểm contour liên tiếp
    perimeter = np.sum([np.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]) for i in range(1, len(xs))])

    # chu vi trên diện tích (độ phức tạp)
    ratio = perimeter / (area + 1e-5)
    return np.concatenate([hu, [ratio, aspect_ratio]])

# ==== KẾT CẤU ==== (LBP thủ công → histogram 10 chiều)
def extract_texture_features(image):
    """
    Trích xuất đặc trưng kết cấu bằng Local Binary Pattern (LBP).
    Tính histogram 10 bins từ mã LBP trên ảnh grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    hist = np.zeros(10)

    # duyệt từng pixel không nằm biên ảnh
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            c = gray[i, j]
            # so sánh pixel lân cận với trung tâm để tạo mã LBP 8 bit
            code = (
                (gray[i-1,j-1] >= c) << 7 |
                (gray[i-1,j  ] >= c) << 6 |
                (gray[i-1,j+1] >= c) << 5 |
                (gray[i  ,j+1] >= c) << 4 |
                (gray[i+1,j+1] >= c) << 3 |
                (gray[i+1,j  ] >= c) << 2 |
                (gray[i+1,j-1] >= c) << 1 |
                (gray[i  ,j-1] >= c)
            )
            bin_idx = min(code * 10 // 256, 9)
            hist[bin_idx] += 1
    return hist / (np.sum(hist) + 1e-7)

# ==== CẠNH LÁ ==== (3 chiều)
def extract_edge_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = (gray < 250).astype(np.uint8)

    # Lấy tọa độ điểm biên (edge points)
    h, w = binary.shape
    edges = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if binary[y, x] == 1:
                neighborhood = binary[y-1:y+2, x-1:x+2]
                if np.any(neighborhood != 1):
                    edges.append((x, y))

    if len(edges) < 3:
        return np.zeros(3)

    edges = np.array(edges)
    xs, ys = edges[:, 0], edges[:, 1]

    # Tính perimeter (độ dài đường viền)
    perimeter = np.sum(np.hypot(np.diff(xs), np.diff(ys)))

    # Tính area xấp xỉ (bằng tổng pixel trong binary)
    area = np.sum(binary)

    # Độ phức tạp cạnh: perimeter / sqrt(area)
    complexity = perimeter / (np.sqrt(area) + 1e-5)

    # Độ nhám (roughness): std của độ dài đoạn giữa các điểm biên liên tiếp
    diffs = np.hypot(np.diff(xs), np.diff(ys))
    edge_roughness = np.std(diffs)

    # Angle variance: biến thiên góc giữa 3 điểm biên liên tiếp
    angles = []
    for i in range(1, len(edges) - 1):
        a = edges[i - 1]
        b = edges[i]
        c = edges[i + 1]
        v1 = a - b
        v2 = c - b
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm > 0:
            angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
            angles.append(angle)
    angle_variance = np.var(angles) if angles else 0

    return np.array([edge_roughness, angle_variance, complexity])

# ==== GÂN LÁ ==== (4 chiều)
def extract_vein_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = (gray < 250).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)

    # Laplacian
    veins = cv2.Laplacian(enhanced, cv2.CV_8U)
    veins = (veins > 30).astype(np.uint8)

    leaf_area = np.sum(mask > 0)
    vein_area = np.sum(veins > 0)
    density = vein_area / (leaf_area + 1e-5)

    # Hướng gân
    sobelx = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1)
    angles = np.arctan2(sobely, sobelx)
    angles = angles[mask > 0]
    direction_consistency = 1.0 - np.std(angles) / np.pi if angles.size else 0

    # Khoảng cách đến gân
    dist = cv2.distanceTransform((1 - veins).astype(np.uint8), cv2.DIST_L2, 3)
    dist = dist[mask > 0]
    mean_width = np.mean(dist) if dist.size else 0
    var_width = np.var(dist) if dist.size else 0

    return np.array([density, direction_consistency, mean_width, var_width])

# ==== TỔNG HỢP ==== (30 + 9 + 10 + 4 + 3= 56 chiều)
def extract_leaf_features(image_rgb):
    color = extract_color_features(image_rgb)
    shape = extract_shape_features(image_rgb)
    texture = extract_texture_features(image_rgb)
    vein = extract_vein_features(image_rgb)
    edge = extract_edge_features(image_rgb)
    return np.concatenate([color, shape, texture, vein, edge])
