"""
沙拉營養分析系統
使用SAM (Segment Anything Model) 和 DPT-Hybrid (MiDaS) 進行圖像分析和營養計算
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import warnings
import time
import sys
import platform
warnings.filterwarnings('ignore')

# 營養數據庫（每100克的營養信息）
NUTRITION_DB = {
    '生菜': {'calories': 15, 'protein': 1.4, 'carbs': 2.9, 'fat': 0.2, 'fiber': 1.3},
    '番茄': {'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'fiber': 1.2},
    '黃瓜': {'calories': 16, 'protein': 0.7, 'carbs': 3.6, 'fat': 0.1, 'fiber': 0.5},
    '胡蘿蔔': {'calories': 41, 'protein': 0.9, 'carbs': 9.6, 'fat': 0.2, 'fiber': 2.8},
    '玉米': {'calories': 96, 'protein': 3.4, 'carbs': 21.3, 'fat': 1.2, 'fiber': 2.4},
    '雞蛋': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0},
    '雞胸肉': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0},
    '堅果': {'calories': 607, 'protein': 20, 'carbs': 21.6, 'fat': 54.4, 'fiber': 7},
    '起司': {'calories': 371, 'protein': 23, 'carbs': 1.3, 'fat': 30, 'fiber': 0},
    '橄欖': {'calories': 115, 'protein': 0.8, 'carbs': 6.0, 'fat': 10.7, 'fiber': 3.2},
    '未知成分': {'calories': 50, 'protein': 2.0, 'carbs': 8.0, 'fat': 1.0, 'fiber': 2.0}
}

class SaladNutritionAnalyzer:
    def _print_environment_info(self):
        """打印環境版本信息"""
        print("\n" + "="*60)
        print("環境信息")
        print("="*60)
        print(f"計算設備: {self.compute_device}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU設備: {torch.cuda.get_device_name(0)}")
            print(f"GPU內存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"Python版本: {sys.version.split()[0]}")
        print(f"操作系統: {platform.system()} {platform.release()}")
        
        # 嘗試獲取其他庫的版本（使用已導入的模塊）
        try:
            print(f"OpenCV版本: {cv2.__version__}")
        except:
            pass
        
        try:
            print(f"NumPy版本: {np.__version__}")
        except:
            pass
        
        try:
            import segment_anything
            print(f"Segment Anything版本: {segment_anything.__version__ if hasattr(segment_anything, '__version__') else '未知'}")
        except:
            pass
        
        print("="*60 + "\n")
    
    def __init__(self, sam_checkpoint_path=None, dpt_model_type='dpt_hybrid', 
                 midas_path=None, result_dir='result'):
        """
        初始化分析器
        Args:
            sam_checkpoint_path: SAM模型檢查點路徑（如果為None，將自動查找）
            dpt_model_type: DPT模型類型（'dpt_hybrid', 'dpt_large', 'dpt_base'）
            midas_path: 本地MiDaS模型資料夾路徑（如果為None，將自動查找）
            result_dir: 結果保存目錄
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_device = 'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'
        
        # 打印環境信息
        self._print_environment_info()
        
        # 創建結果資料夾
        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        print(f"結果將保存到 '{self.result_dir}' 資料夾")
        
        # 初始化SAM模型
        print("正在載入SAM模型...")
        try:
            # 自動查找SAM模型文件
            if sam_checkpoint_path is None:
                possible_paths = [
                    "sam_vit_h_4b8939.pth",
                    "sam_vit_h.pth",
                    "./sam_vit_h_4b8939.pth",
                    "./sam_vit_h.pth"
                ]
                sam_checkpoint_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        sam_checkpoint_path = path
                        print(f"找到SAM模型: {path}")
                        break
                
                if sam_checkpoint_path is None:
                    print("未找到SAM模型文件，請確保模型文件在當前目錄")
                    print("從以下網址下載: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
                    raise FileNotFoundError("SAM模型文件未找到")
            
            self.sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
            self.sam_model.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam_model)
            # 創建自動mask生成器（用於更好的分割）
            self.sam_mask_generator = SamAutomaticMaskGenerator(
                self.sam_model,
                points_per_side=32,  # 增加採樣點密度
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # 最小區域面積（像素）
            )
            print("✓ SAM模型載入完成")
        except Exception as e:
            print(f"✗ SAM模型載入失敗: {e}")
            print("將使用簡化版分割功能")
            self.sam_predictor = None
        
        # 初始化DPT模型
        print("正在載入DPT-Hybrid模型...")
        try:
            # 自動查找本地MiDaS資料夾
            if midas_path is None:
                possible_midas_paths = [
                    "D:\\AI_CODE\\isl-org-MiDaS-4545977",
                    "./isl-org-MiDaS-4545977",
                    "../isl-org-MiDaS-4545977",
                    "isl-org-MiDaS-4545977"
                ]
                midas_path = None
                for path in possible_midas_paths:
                    if os.path.exists(path) and os.path.exists(os.path.join(path, "hubconf.py")):
                        midas_path = path
                        print(f"找到本地MiDaS模型: {path}")
                        break
            
            if midas_path and os.path.exists(midas_path):
                # 從本地MiDaS資料夾載入
                if midas_path not in sys.path:
                    sys.path.insert(0, midas_path)
                
                # 載入hubconf並獲取模型函數
                import importlib.util
                hubconf_path = os.path.join(midas_path, "hubconf.py")
                spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
                hubconf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(hubconf)
                
                # 根據模型類型選擇對應的函數
                if dpt_model_type == 'dpt_hybrid':
                    self.dpt_model = hubconf.DPT_Hybrid(pretrained=True)
                elif dpt_model_type == 'dpt_large':
                    self.dpt_model = hubconf.DPT_Large(pretrained=True)
                elif dpt_model_type == 'dpt_beit_large_384':
                    self.dpt_model = hubconf.DPT_BEiT_L_384(pretrained=True)
                elif dpt_model_type == 'dpt_beit_base_384':
                    self.dpt_model = hubconf.DPT_BEiT_B_384(pretrained=True)
                else:
                    # 默認使用DPT_Hybrid
                    self.dpt_model = hubconf.DPT_Hybrid(pretrained=True)
                
                self.dpt_model.to(self.device)
                self.dpt_model.eval()
                print(f"✓ DPT模型載入完成（從本地: {midas_path}）")
            else:
                # 如果找不到本地模型，使用torch.hub從網上下載
                print("未找到本地MiDaS模型，將從網上下載...")
                model_map = {
                    'dpt_hybrid': 'DPT_Hybrid',
                    'dpt_large': 'DPT_Large',
                    'dpt_beit_large_384': 'DPT_BEiT_L_384',
                    'dpt_beit_base_384': 'DPT_BEiT_B_384'
                }
                hub_function = model_map.get(dpt_model_type, 'DPT_Hybrid')
                self.dpt_model = torch.hub.load('intel-isl/MiDaS', hub_function)
                self.dpt_model.to(self.device)
                self.dpt_model.eval()
                print("✓ DPT模型載入完成（從網路下載）")
        except Exception as e:
            print(f"✗ DPT模型載入失敗: {e}")
            import traceback
            traceback.print_exc()
            self.dpt_model = None
    
    def load_image(self, image_path):
        """載入圖像"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"無法載入圖像: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = np.array(image_path)
            if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 4:
                image_rgb = image_rgb[:, :, :3]
        
        # 保存原始圖像
        self.save_image(image_rgb, '01_original_image.jpg')
        
        return image_rgb
    
    def save_image(self, image, filename):
        """保存圖像到result資料夾"""
        filepath = os.path.join(self.result_dir, filename)
        if len(image.shape) == 2 or image.shape[2] == 1:
            # 灰度圖
            cv2.imwrite(filepath, image)
        else:
            # RGB圖像，需要轉換為BGR
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, image_bgr)
        print(f"  已保存: {filepath}")
    
    def segment_image_with_sam(self, image_rgb):
        """
        使用SAM進行圖像分割（改進版：使用自動mask生成）
        """
        if self.sam_predictor is None:
            # 簡化版分割：使用顏色和紋理特徵
            return self.simplified_segmentation(image_rgb)
        
        try:
            h, w = image_rgb.shape[:2]
            
            # 使用自動mask生成器來獲取所有可能的物體mask
            print("  正在使用SAM自動mask生成...")
            if hasattr(self, 'sam_mask_generator'):
                # 使用自動mask生成器
                sam_start = time.time()
                masks = self.sam_mask_generator.generate(image_rgb)
                sam_time = time.time() - sam_start
                print(f"  SAM推理時間: {sam_time:.2f} 秒")
            else:
                # 備用方案：使用更密集的點採樣
                self.sam_predictor.set_image(image_rgb)
                
                # 更密集的網格採樣（5x5 = 25個點）
                step_h, step_w = h // 5, w // 5
                all_masks_data = []
                
                for i in range(1, 6):
                    for j in range(1, 6):
                        point_coords = np.array([[step_h * i, step_w * j]])
                        point_labels = np.array([1])
                        
                        masks_pred, scores, logits = self.sam_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        
                        # 保存所有mask（不只是最好的）
                        for mask_idx, mask in enumerate(masks_pred):
                            if scores[mask_idx] > 0.7:  # 只保留高分的mask
                                all_masks_data.append({
                                    'segmentation': mask,
                                    'stability_score': scores[mask_idx],
                                    'area': mask.sum()
                                })
                
                masks = all_masks_data
            
            if not masks:
                print("  警告: 未生成任何mask，使用備用分割方法")
                return self.simplified_segmentation(image_rgb)
            
            # 過濾和整理masks
            print(f"  生成了 {len(masks)} 個候選mask")
            
            # 按穩定性分數排序，保留最好的masks
            if isinstance(masks[0], dict):
                masks = sorted(masks, key=lambda x: x.get('stability_score', x.get('predicted_iou', 0)), reverse=True)
                # 只保留前50個最好的masks
                masks = masks[:50]
                mask_segmentations = [m['segmentation'] for m in masks]
            else:
                mask_segmentations = masks
            
            # 去重和合併重疊的masks
            segments = []
            used_mask = np.zeros((h, w), dtype=bool)
            segment_vis = image_rgb.copy().astype(np.float32)
            colors = plt.cm.Set3(np.linspace(0, 1, len(mask_segmentations)))
            
            # 保存所有原始masks的可視化
            mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
            for mask in mask_segmentations[:20]:  # 只顯示前20個
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                mask_vis[mask_bool] = [255, 255, 255]
            self.save_image(mask_vis, '02_sam_all_masks.jpg')
            
            # 選擇獨立的、不重疊的區域
            for idx, mask in enumerate(mask_segmentations):
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                mask_area = mask_bool.sum()
                
                # 過濾太小或太大的區域
                if mask_area < 100 or mask_area > (h * w * 0.8):
                    continue
                
                # 計算與已使用區域的重疊度
                overlap = np.logical_and(used_mask, mask_bool).sum()
                overlap_ratio = overlap / mask_area if mask_area > 0 else 0
                
                # 如果重疊度小於30%，認為是新的獨立區域
                if overlap_ratio < 0.3:
                    segments.append(mask_bool)
                    used_mask = np.logical_or(used_mask, mask_bool)
                    
                    # 為可視化添加顏色（半透明覆蓋）
                    color = np.array(colors[idx][:3]) * 255
                    segment_vis[mask_bool] = segment_vis[mask_bool] * 0.6 + color * 0.4
            
            print(f"  最終識別到 {len(segments)} 個獨立區域")
            
            # 為每個分割區域繪製藍色邊緣線（沿著實際輪廓）
            segment_vis_with_boxes = segment_vis.copy()
            for segment_mask in segments:
                # 將mask轉換為uint8格式用於輪廓檢測
                mask_uint8 = (segment_mask.astype(np.uint8) * 255)
                
                # 找到mask的輪廓
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 繪製藍色輪廓線
                edge_thickness = max(2, int(min(h, w) / 400))  # 根據圖像大小調整線條粗細
                
                # 轉換為BGR格式以便使用cv2.drawContours
                segment_vis_bgr = cv2.cvtColor(segment_vis_with_boxes.astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # 繪製藍色輪廓（BGR格式中藍色是(255, 0, 0)）
                cv2.drawContours(segment_vis_bgr, contours, -1, (255, 0, 0), edge_thickness)
                
                # 轉回RGB格式
                segment_vis_with_boxes = cv2.cvtColor(segment_vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            # 保存分割結果（帶藍色邊框）
            if len(segments) > 0:
                self.save_image(segment_vis_with_boxes.astype(np.uint8), '03_sam_segmentation_result.jpg')
            else:
                # 如果沒有找到獨立區域，使用第一個mask
                if mask_segmentations:
                    mask_bool = mask_segmentations[0].astype(bool) if mask_segmentations[0].dtype != bool else mask_segmentations[0]
                    segments.append(mask_bool)
                    segment_vis = image_rgb.copy().astype(np.float32)
                    segment_vis[mask_bool] = segment_vis[mask_bool] * 0.7 + np.array([0, 255, 0]) * 0.3
                    
                    # 繪製藍色邊緣線（沿著實際輪廓）
                    mask_uint8 = (mask_bool.astype(np.uint8) * 255)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    edge_thickness = max(2, int(min(h, w) / 400))
                    
                    # 轉換為BGR格式以便使用cv2.drawContours
                    segment_vis_bgr = cv2.cvtColor(segment_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    cv2.drawContours(segment_vis_bgr, contours, -1, (255, 0, 0), edge_thickness)
                    segment_vis = cv2.cvtColor(segment_vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                    
                    self.save_image(segment_vis.astype(np.uint8), '03_sam_segmentation_result.jpg')
            
            return segments if segments else [np.ones((h, w), dtype=bool)]
            
        except Exception as e:
            print(f"SAM分割錯誤: {e}")
            import traceback
            traceback.print_exc()
            return self.simplified_segmentation(image_rgb)
    
    def simplified_segmentation(self, image_rgb):
        """簡化版分割（當SAM不可用時）"""
        # 轉換為HSV色彩空間
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # 使用K-means聚類進行分割
        from sklearn.cluster import KMeans
        
        # 重塑圖像
        pixels = image_rgb.reshape(-1, 3)
        
        # K-means聚類（假設沙拉有5-7種主要成分）
        n_clusters = 6
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        labels = labels.reshape(image_rgb.shape[:2])
        
        # 為每個聚類創建mask
        segments = []
        segment_vis = image_rgb.copy().astype(np.float32)
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        for i in range(n_clusters):
            mask = (labels == i).astype(np.uint8)
            if np.sum(mask) > 500:  # 過濾小區域
                segments.append(mask.astype(bool))
                # 為可視化添加顏色
                color = np.array(colors[i][:3]) * 255
                segment_vis[mask > 0] = segment_vis[mask > 0] * 0.6 + color * 0.4
        
        # 為每個分割區域繪製藍色邊緣線（沿著實際輪廓）
        h, w = image_rgb.shape[:2]
        # 轉換為BGR格式以便使用cv2.drawContours
        segment_vis_bgr = cv2.cvtColor(segment_vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        for segment_mask in segments:
            # 將mask轉換為uint8格式用於輪廓檢測
            mask_uint8 = (segment_mask.astype(np.uint8) * 255)
            
            # 找到mask的輪廓
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 繪製藍色輪廓線
            edge_thickness = max(2, int(min(h, w) / 400))
            cv2.drawContours(segment_vis_bgr, contours, -1, (255, 0, 0), edge_thickness)
        
        # 轉回RGB格式
        segment_vis = cv2.cvtColor(segment_vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        # 保存分割結果（帶藍色邊框）
        self.save_image(segment_vis.astype(np.uint8), '03_kmeans_segmentation_result.jpg')
        
        return segments if segments else [np.ones(image_rgb.shape[:2], dtype=bool)]
    
    def estimate_depth_with_dpt(self, image_rgb):
        """
        使用DPT-Hybrid估計深度圖
        """
        if self.dpt_model is None:
            # 返回假設的深度圖
            h, w = image_rgb.shape[:2]
            depth = np.ones((h, w), dtype=np.float32)
            # 保存假設的深度圖
            depth_vis = (depth * 255).astype(np.uint8)
            self.save_image(depth_vis, '04_dpt_depth_map.jpg')
            return depth
        
        try:
            # 轉換為torch tensor
            img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 調整大小以符合模型輸入
            img_tensor = F.interpolate(
                img_tensor, 
                size=(384, 384), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 預測深度
            dpt_start = time.time()
            with torch.no_grad():
                depth = self.dpt_model(img_tensor)
                depth = F.interpolate(
                    depth.unsqueeze(1),
                    size=image_rgb.shape[:2],
                    mode='bilinear',
                    align_corners=False
                ).squeeze().cpu().numpy()
            dpt_time = time.time() - dpt_start
            print(f"  DPT推理時間: {dpt_time:.2f} 秒")
            
            # 正規化深度值
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            # 保存深度圖（可視化版本）
            depth_vis = (depth * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_VIRIDIS)
            depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            self.save_image(depth_colored_rgb, '04_dpt_depth_map.jpg')
            
            return depth
            
        except Exception as e:
            print(f"DPT深度估計錯誤: {e}")
            h, w = image_rgb.shape[:2]
            depth = np.ones((h, w), dtype=np.float32)
            depth_vis = (depth * 255).astype(np.uint8)
            self.save_image(depth_vis, '04_dpt_depth_map.jpg')
            return depth
    
    def estimate_volume(self, mask, depth_map, pixel_to_cm_ratio=0.1):
        """
        估算每個分割區域的體積
        Args:
            mask: 分割mask
            depth_map: 深度圖
            pixel_to_cm_ratio: 像素到厘米的轉換比例（需要根據實際情況調整）
        """
        # 計算mask區域的面積（像素數）
        area_pixels = np.sum(mask)
        
        # 計算該區域的平均深度
        region_depth = depth_map[mask]
        avg_depth = np.mean(region_depth) if len(region_depth) > 0 else 1.0
        
        # 估算體積（假設為圓柱體）
        # 面積轉換為cm²
        area_cm2 = area_pixels * (pixel_to_cm_ratio ** 2)
        
        # 深度轉換為cm（假設深度範圍在0-10cm之間）
        depth_cm = avg_depth * 10
        
        # 體積 = 面積 * 深度
        volume_cm3 = area_cm2 * depth_cm
        
        # 轉換為重量（假設密度約為1 g/cm³）
        weight_grams = volume_cm3 * 1.0
        
        return weight_grams
    
    def identify_food_component(self, image_region):
        """
        識別食物成分（簡化版：基於顏色特徵）
        實際應用中應該使用更先進的圖像分類模型
        """
        # 確保圖像區域有效
        if image_region.size == 0 or len(image_region.shape) < 2:
            return '未知成分'
        
        # 重塑為2D數組以便計算平均顏色
        if len(image_region.shape) == 3:
            pixels = image_region.reshape(-1, 3)
        else:
            pixels = image_region.reshape(-1, 1)
        
        # 過濾掉黑色/無效像素（可能是背景）
        if len(pixels.shape) == 2 and pixels.shape[1] == 3:
            valid_pixels = pixels[(pixels.sum(axis=1) > 10)]  # 過濾接近黑色的像素
            if len(valid_pixels) == 0:
                return '未知成分'
            avg_color = np.mean(valid_pixels, axis=0)
        else:
            return '未知成分'
        
        # 將平均顏色轉換為HSV
        # 創建一個1x1的RGB圖像來轉換HSV
        avg_color_uint8 = np.clip(avg_color, 0, 255).astype(np.uint8)
        color_rgb = avg_color_uint8.reshape(1, 1, 3)
        color_hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = color_hsv[0, 0]
        
        # 基於顏色特徵的簡單分類
        if v < 50:  # 很暗
            return '未知成分'
        elif h < 15 or h > 165:  # 紅色/黃色範圍
            if avg_color[0] > 200 and avg_color[1] < 100:  # 很紅，低綠
                return '番茄'
            elif avg_color[0] > 150 and avg_color[1] > 100:  # 橙色
                return '胡蘿蔔'
            else:
                return '未知成分'
        elif 15 < h < 45:  # 黃色範圍
            if avg_color[1] > 150:  # 高綠色值（黃色）
                return '玉米'
            else:
                return '未知成分'
        elif 45 < h < 75:  # 綠色範圍
            if s < 80 and v > 150:  # 低飽和度、高亮度（淺綠，可能是生菜）
                return '生菜'
            elif s > 80:  # 高飽和度（深綠，可能是黃瓜或其他綠葉蔬菜）
                return '黃瓜'
            else:
                return '生菜'
        elif 100 < h < 130:  # 藍綠色範圍（可能是某些特殊蔬菜）
            return '未知成分'
        else:
            return '未知成分'
    
    def calculate_nutrition(self, segments, image_rgb, depth_map):
        """
        計算營養數據
        會自動合併相同類型的成分，避免重複計算
        """
        total_nutrition = {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fat': 0,
            'fiber': 0,
            'weight': 0
        }
        
        # 先用字典來累加相同類型的成分
        component_dict = {}
        
        for i, segment_mask in enumerate(segments):
            # 提取該區域的圖像
            masked_image = image_rgb.copy()
            masked_image[~segment_mask] = 0
            region_image = masked_image[
                np.any(segment_mask, axis=1)
            ][:, np.any(segment_mask, axis=0)]
            
            if region_image.size == 0:
                continue
            
            # 識別食物成分
            food_type = self.identify_food_component(region_image)
            
            # 估算重量
            weight = self.estimate_volume(segment_mask, depth_map)
            
            # 計算營養數據
            nutrition_per_100g = NUTRITION_DB.get(food_type, NUTRITION_DB['未知成分'])
            
            # 如果該類型已存在，累加重量和營養值
            if food_type in component_dict:
                component_dict[food_type]['weight_g'] += weight
                component_dict[food_type]['calories'] += weight * nutrition_per_100g['calories'] / 100
                component_dict[food_type]['protein'] += weight * nutrition_per_100g['protein'] / 100
                component_dict[food_type]['carbs'] += weight * nutrition_per_100g['carbs'] / 100
                component_dict[food_type]['fat'] += weight * nutrition_per_100g['fat'] / 100
                component_dict[food_type]['fiber'] += weight * nutrition_per_100g['fiber'] / 100
            else:
                # 創建新的成分記錄
                component_dict[food_type] = {
                    'type': food_type,
                    'weight_g': weight,
                    'calories': weight * nutrition_per_100g['calories'] / 100,
                    'protein': weight * nutrition_per_100g['protein'] / 100,
                    'carbs': weight * nutrition_per_100g['carbs'] / 100,
                    'fat': weight * nutrition_per_100g['fat'] / 100,
                    'fiber': weight * nutrition_per_100g['fiber'] / 100,
                }
        
        # 轉換為列表格式
        component_details = list(component_dict.values())
        
        # 按重量排序（從大到小）
        component_details.sort(key=lambda x: x['weight_g'], reverse=True)
        
        # 計算總營養數據
        for comp in component_details:
            total_nutrition['weight'] += comp['weight_g']
            total_nutrition['calories'] += comp['calories']
            total_nutrition['protein'] += comp['protein']
            total_nutrition['carbs'] += comp['carbs']
            total_nutrition['fat'] += comp['fat']
            total_nutrition['fiber'] += comp['fiber']
        
        return total_nutrition, component_details
    
    def analyze_salad(self, image_path, visualize=True):
        """
        分析沙拉圖像並返回營養數據
        Args:
            image_path: 圖像路徑或PIL Image對象
            visualize: 是否顯示可視化結果
        Returns:
            營養數據字典和成分詳情
        """
        total_start_time = time.time()
        
        print("\n" + "="*60)
        print(f"正在分析圖像: {image_path}")
        print("="*60)
        
        # 載入圖像
        print("\n[步驟 1/4] 載入原始圖像...")
        step_start = time.time()
        image_rgb = self.load_image(image_path)
        step_time = time.time() - step_start
        print(f"  耗時: {step_time:.2f} 秒")
        
        # 圖像分割
        print("\n[步驟 2/4] 使用SAM進行圖像分割...")
        step_start = time.time()
        segments = self.segment_image_with_sam(image_rgb)
        step_time = time.time() - step_start
        print(f"✓ 識別到 {len(segments)} 個成分區域")
        print(f"  總耗時: {step_time:.2f} 秒")
        
        # 深度估計
        print("\n[步驟 3/4] 使用DPT-Hybrid估計深度...")
        step_start = time.time()
        depth_map = self.estimate_depth_with_dpt(image_rgb)
        step_time = time.time() - step_start
        print("✓ 深度圖已生成")
        print(f"  總耗時: {step_time:.2f} 秒")
        
        # 計算營養數據
        print("\n[步驟 4/4] 計算營養數據...")
        step_start = time.time()
        total_nutrition, component_details = self.calculate_nutrition(
            segments, image_rgb, depth_map
        )
        step_time = time.time() - step_start
        print("✓ 營養數據計算完成")
        print(f"  耗時: {step_time:.2f} 秒")
        
        # 可視化
        if visualize:
            print("\n正在生成最終可視化結果...")
            step_start = time.time()
            self.visualize_results(image_rgb, segments, depth_map, total_nutrition, component_details)
            step_time = time.time() - step_start
            print(f"  耗時: {step_time:.2f} 秒")
        
        total_time = time.time() - total_start_time
        
        print(f"\n{'='*60}")
        print(f"所有處理結果已保存到 '{self.result_dir}' 資料夾")
        print("="*60)
        print(f"\n⏱️  總推理時間: {total_time:.2f} 秒 ({total_time/60:.2f} 分鐘)")
        print(f"📊 計算設備: {self.compute_device}")
        print(f"📈 處理速度: {1/total_time:.2f} 圖像/秒")
        print("="*60 + "\n")
        
        return total_nutrition, component_details
    
    def visualize_results(self, image_rgb, segments, depth_map, total_nutrition, component_details):
        """可視化結果（支持中文顯示）"""
        # 設置中文字體
        # Windows 系統常用中文字體
        chinese_fonts = [
            'Microsoft YaHei',      # 微軟雅黑
            'SimHei',                # 黑體
            'SimSun',                # 宋體
            'KaiTi',                 # 楷體
            'FangSong',              # 仿宋
            'Arial Unicode MS',      # Arial Unicode MS
        ]
        
        # 嘗試設置中文字體
        font_set = False
        for font_name in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
                # 測試字體是否可用
                fig_test = plt.figure(figsize=(1, 1))
                ax_test = fig_test.add_subplot(111)
                ax_test.text(0.5, 0.5, '測試', fontsize=10)
                plt.close(fig_test)
                font_set = True
                print(f"  使用字體: {font_name}")
                break
            except:
                continue
        
        if not font_set:
            print("  ⚠️ 警告: 無法設置中文字體，中文可能顯示為方框")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        
        # 創建圖形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('沙拉營養分析報告', fontsize=18, fontweight='bold', y=0.98)
        
        # 原始圖像
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('原始圖像', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 分割結果（帶藍色邊框）
        segmented_image = image_rgb.copy().astype(np.float32)
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        h, w = image_rgb.shape[:2]
        
        for i, segment_mask in enumerate(segments):
            # 添加半透明顏色覆蓋
            segmented_image[segment_mask] = segmented_image[segment_mask] * 0.6 + \
                                            np.array(colors[i][:3]) * 255 * 0.4
            
            # 繪製藍色邊緣線（沿著實際輪廓）
            mask_uint8 = (segment_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge_thickness = max(2, int(min(h, w) / 400))
            
            # 轉換為BGR格式以便使用cv2.drawContours
            segmented_image_bgr = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.drawContours(segmented_image_bgr, contours, -1, (255, 0, 0), edge_thickness)
            segmented_image = cv2.cvtColor(segmented_image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        axes[0, 1].imshow(segmented_image.astype(np.uint8))
        axes[0, 1].set_title(f'SAM分割結果 ({len(segments)} 個區域)', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 深度圖
        im = axes[1, 0].imshow(depth_map, cmap='viridis')
        axes[1, 0].set_title('DPT-Hybrid 深度圖', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # 營養數據
        axes[1, 1].axis('off')
        axes[1, 1].set_facecolor('#f0f0f0')
        
        # 構建營養數據文本
        nutrition_text = f"""總營養數據
{'='*30}
總重量: {total_nutrition['weight']:.1f} 克

熱量: {total_nutrition['calories']:.1f} 大卡
蛋白質: {total_nutrition['protein']:.1f} 克
碳水化合物: {total_nutrition['carbs']:.1f} 克
脂肪: {total_nutrition['fat']:.1f} 克
纖維: {total_nutrition['fiber']:.1f} 克

成分詳情:"""
        
        for comp in component_details:
            nutrition_text += f"\n• {comp['type']}: {comp['weight_g']:.1f}克 ({comp['calories']:.1f}大卡)"
        
        # 使用支持中文的字體顯示文本（移除 monospace，使用中文字體）
        axes[1, 1].text(0.05, 0.95, nutrition_text, fontsize=11, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        family=plt.rcParams['font.sans-serif'][0] if font_set else 'sans-serif')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 為標題留出空間
        result_path = os.path.join(self.result_dir, '05_final_analysis_result.png')
        plt.savefig(result_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 最終結果已保存到: {result_path}")
        plt.close()  # 關閉圖形以避免顯示（如果需要顯示可以改為plt.show()）


def main():
    """主函數"""
    print("=" * 50)
    print("沙拉營養分析系統")
    print("=" * 50)
    
    # 初始化分析器（會自動查找SAM模型文件）
    analyzer = SaladNutritionAnalyzer(sam_checkpoint_path=None)
    
    # 示例：分析圖像
    # 請替換為您的沙拉圖像路徑
    image_path = input("請輸入沙拉圖像路徑（或按Enter使用測試模式）: ").strip()
    
    if not image_path:
        print("未提供圖像路徑，請在代碼中指定圖像路徑")
        return
    
    # 執行分析
    try:
        total_nutrition, component_details = analyzer.analyze_salad(image_path)
        
        # 打印結果
        print("\n" + "=" * 50)
        print("營養分析結果")
        print("=" * 50)
        print(f"總重量: {total_nutrition['weight']:.1f} 克")
        print(f"總熱量: {total_nutrition['calories']:.1f} 大卡")
        print(f"蛋白質: {total_nutrition['protein']:.1f} 克")
        print(f"碳水化合物: {total_nutrition['carbs']:.1f} 克")
        print(f"脂肪: {total_nutrition['fat']:.1f} 克")
        print(f"纖維: {total_nutrition['fiber']:.1f} 克")
        print("\n成分詳情:")
        for comp in component_details:
            print(f"  - {comp['type']}: {comp['weight_g']:.1f}克 "
                  f"({comp['calories']:.1f}大卡)")
        
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

