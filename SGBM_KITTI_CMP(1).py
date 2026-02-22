import cv2
import numpy as np
import os

# --- 1. 路径配置 ---
base_path = r"D:\datasets\kitti_2015\training"
img_l_dir = os.path.join(base_path, "image_2")
img_r_dir = os.path.join(base_path, "image_3")
gt_dir = os.path.join(base_path, "disp_noc_0")

output_dir = os.path.join(base_path, "output_kitti_SOTA_SGBM")
os.makedirs(output_dir, exist_ok=True)

# --- 2. 辅助函数 ---
def read_kitti_disp(filename):
    disp_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if disp_img is None: return None
    disp = disp_img.astype(np.float32) / 256.0
    valid_mask = disp_img > 0
    return disp, valid_mask

# 注意：去掉了原本的 fill_left_side_holes 函数

def visualize_disparity_turbo(disp, max_disp=192):
    disp_clean = np.maximum(disp, 0)
    disp_norm = np.clip((disp_clean / max_disp) * 255, 0, 255).astype(np.uint8)
    try: colormap = cv2.COLORMAP_TURBO
    except AttributeError: colormap = cv2.COLORMAP_JET 
    return cv2.applyColorMap(disp_norm, colormap)

def visualize_error_kitti_style(disp_est, disp_gt, valid_mask):
    h, w = disp_est.shape
    abs_error = np.abs(disp_est - disp_gt)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = abs_error / (disp_gt + 1e-6)
        rel_error[~np.isfinite(rel_error)] = 0
    # D1 指标：绝对误差 > 3像素 且 相对误差 > 5%
    is_bad = (abs_error > 3.0) & (rel_error > 0.05) & valid_mask
    error_img = np.zeros((h, w, 3), dtype=np.uint8)
    error_img[valid_mask] = [255, 255, 255]
    error_img[is_bad] = [0, 0, 255]
    return error_img, is_bad

# --- 3. 核心参数优化 ---
W_SIZE = 7
MAX_D  = 128

params = {
    "minDisparity": 0,
    "numDisparities": MAX_D,
    "blockSize": W_SIZE,
    "P1": 4 * 3 * W_SIZE**2,   
    "P2": 32 * 3 * W_SIZE**2,  
    "disp12MaxDiff": 0,         # 必须为 0，让 WLS 接管一致性检查
    "uniquenessRatio": 10,      
    "speckleWindowSize": 50,  
    "speckleRange": 2,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY 
    # "mode": cv2.STEREO_SGBM_MODE_HH 
}

left_matcher = cv2.StereoSGBM_create(**params)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# 优化 WLS 滤波器参数
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(4000.0) 
wls_filter.setSigmaColor(1.2) 

# --- 4. 主循环 ---
image_files = sorted([f for f in os.listdir(img_l_dir) if f.endswith('_10.png')])
all_stats = []

for file_name in image_files:
    imgL = cv2.imread(os.path.join(img_l_dir, file_name))
    imgR = cv2.imread(os.path.join(img_r_dir, file_name))
    
    gt_res = read_kitti_disp(os.path.join(gt_dir, file_name))
    if gt_res is None: continue
    disp_gt, valid_mask_gt = gt_res

    # ✨ 核心修复：边缘填充欺骗法 (Border Padding)
    # 在左侧填充 MAX_D 宽度的边界，模式为 BORDER_REPLICATE (复制边缘像素)
    imgL_padded = cv2.copyMakeBorder(imgL, 0, 0, MAX_D, 0, cv2.BORDER_REPLICATE)
    imgR_padded = cv2.copyMakeBorder(imgR, 0, 0, MAX_D, 0, cv2.BORDER_REPLICATE)

    # A. 在填充后的图像上计算原始视差
    disp_l_raw_padded = left_matcher.compute(imgL_padded, imgR_padded)
    disp_r_raw_padded = right_matcher.compute(imgR_padded, imgL_padded)

    # B. 对填充后的视差图应用 WLS 滤波
    disp_filtered_padded = wls_filter.filter(disp_l_raw_padded, imgL_padded, None, disp_r_raw_padded)

    # C. ✨ 核心修复：裁切掉之前填充的假边缘
    # 将最左侧 MAX_D 宽度的数据切除，恢复到原始图像尺寸
    disp_filtered = disp_filtered_padded[:, MAX_D:]

    # D. 转换为浮点型真实视差 (除以 16)
    disp_final = disp_filtered.astype(np.float32) / 16.0

    # E. 评估与可视化
    error_img, bad_mask = visualize_error_kitti_style(disp_final, disp_gt, valid_mask_gt)
    num_valid = np.sum(valid_mask_gt)
    if num_valid > 0:
        bad_ratio = (np.sum(bad_mask) / num_valid) * 100
        all_stats.append(bad_ratio)
        print(f"{file_name:<20} | D1 Error: {bad_ratio:>6.2f}%")

    color_disp = visualize_disparity_turbo(disp_final, max_disp=MAX_D)
    cv2.imwrite(os.path.join(output_dir, f"disp_turbo_{file_name}"), color_disp)
    cv2.imwrite(os.path.join(output_dir, f"error_{file_name}"), error_img)

if all_stats:
    print("-" * 35)
    print(f"KITTI 平均 D1 误差: {np.mean(all_stats):.2f}%")