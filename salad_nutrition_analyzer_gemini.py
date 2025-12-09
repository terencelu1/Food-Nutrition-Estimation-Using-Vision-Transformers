"""
沙拉營養分析系統（Gemini 增強版）
使用 Gemini Vision API + SAM + DPT-Hybrid 進行更精準的圖像分析和營養計算
"""

import json
import numpy as np
import base64
import requests
from pathlib import Path
from salad_nutrition_analyzer import SaladNutritionAnalyzer, NUTRITION_DB

GEMINI_AVAILABLE = True  # 使用 REST API，不需要 google-generativeai 套件


class GeminiSaladAnalyzer(SaladNutritionAnalyzer):
    """
    增強版沙拉分析器，整合 Gemini Vision API
    """
    
    def __init__(self, sam_checkpoint_path=None, dpt_model_type='dpt_hybrid', 
                 midas_path=None, result_dir='result',
                 use_gemini=True, gemini_api_key=None):
        """
        初始化分析器
        Args:
            sam_checkpoint_path: SAM模型檢查點路徑
            dpt_model_type: DPT模型類型
            midas_path: 本地MiDaS模型資料夾路徑
            result_dir: 結果保存目錄
            use_gemini: 是否使用 Gemini API
            gemini_api_key: Gemini API 密鑰
        """
        # 初始化父類
        super().__init__(sam_checkpoint_path, dpt_model_type, midas_path, result_dir)
        
        self.use_gemini = use_gemini and GEMINI_AVAILABLE
        self.gemini_analysis = None
        
        if self.use_gemini:
            if gemini_api_key is None:
                # 從環境變量讀取 API key
                import os
                gemini_api_key = os.getenv('GEMINI_API_KEY')
                if gemini_api_key is None:
                    print("⚠️ 警告: 未提供 Gemini API Key")
                    print("   請設置環境變量 GEMINI_API_KEY 或在初始化時傳入 gemini_api_key 參數")
                    self.use_gemini = False
                    return
            
            self.gemini_api_key = gemini_api_key
            # 直接使用 Gemini 2.5 Flash-Lite
            self.gemini_model_name = "gemini-2.5-flash-lite"
            # 使用 v1beta API（因為 v1 可能不支持所有模型）
            self.gemini_api_version = "v1beta"
            self.gemini_api_url = f"https://generativelanguage.googleapis.com/{self.gemini_api_version}/models/{self.gemini_model_name}:generateContent"
            print(f"✓ Gemini API 已配置（使用 REST API，模型: {self.gemini_model_name}）")
    
    def analyze_with_gemini(self, image_path):
        """
        使用 Gemini Vision API 進行深度分析
        """
        if not self.use_gemini:
            return None
        
        try:
            # 讀取圖像並轉換為 base64
            image_data = Path(image_path).read_bytes()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 設計精準的 prompt
            prompt = """
你是一位專業的營養分析師和計算機視覺專家。請仔細分析這張沙拉圖片，提供以下結構化的分析結果（必須是有效的JSON格式，不要有任何額外的文字說明）：

{
  "reference_analysis": {
    "detected_objects": [
      {
        "object": "盤子",
        "estimated_diameter_cm": 25.0,
        "confidence": 0.9,
        "reasoning": "標準沙拉盤，直徑約25cm"
      }
    ],
    "pixel_to_cm_ratio_estimate": 0.035,
    "ratio_confidence": 0.85,
    "calculation_method": "基於盤子直徑和圖像像素數計算"
  },
  
  "component_analysis": [
    {
      "name": "玉米",
      "visual_description": "黃色玉米粒",
      "volume_ratio": 0.30,
      "estimated_weight_g": 100,
      "confidence": 0.9,
      "spatial_distribution": "分散",
      "depth_estimate_cm": 1.5,
      "density_factor": 0.65
    }
  ],
  
  "depth_analysis": {
    "overall_average_depth_cm": 2.5,
    "max_depth_cm": 4.0,
    "min_depth_cm": 1.0,
    "depth_distribution": "不均勻，中心較厚",
    "packing_density": 0.7
  },
  
  "calibration_factors": {
    "overall_volume_correction": 0.6,
    "weight_correction": 0.5,
    "reasoning": "考慮空隙、實際密度和形狀因素"
  },
  
  "nutrition_estimate": {
    "total_weight_g_range": [250, 350],
    "total_calories_range": [280, 420],
    "confidence": 0.8
  },
  
  "quality_assessment": {
    "image_quality": "good",
    "lighting_conditions": "adequate",
    "angle_perspective": "top-down",
    "occlusion_level": "low"
  }
}

重要要求：
1. 所有數值必須基於視覺觀察和常識判斷
2. volume_ratio 的總和應該接近 1.0
3. 重量估算要保守（寧可低估）
4. 考慮食物的實際密度（生菜很輕約0.15，玉米較重約0.65）
5. 考慮空隙和堆疊效應
6. 如果無法確定某個成分，confidence 設為較低值
7. 必須返回有效的 JSON，不要有任何額外的文字說明或 markdown 代碼塊
8. 直接返回 JSON 對象，不要用 ```json 包裹
"""
            
            print(f"  正在調用 Gemini Vision API (模型: {self.gemini_model_name})...")
            
            # 使用 REST API 調用 Gemini
            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_base64
                            }
                        }
                    ]
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            params = {
                "key": self.gemini_api_key
            }
            
            # 直接使用 Gemini 2.5 Flash-Lite
            try:
                response = requests.post(
                    self.gemini_api_url,
                    json=payload,
                    headers=headers,
                    params=params,
                    timeout=60
                )
                
                if response.status_code != 200:
                    # 處理錯誤
                    try:
                        error_data = response.json() if response.text else {}
                        error_msg = error_data.get('error', {}).get('message', response.text[:500])
                    except:
                        error_msg = response.text[:500] if response.text else "未知錯誤"
                    print(f"⚠️ Gemini API 請求失敗: {response.status_code}")
                    print(f"錯誤信息: {error_msg}")
                    return None
                    
            except Exception as e:
                print(f"⚠️ Gemini API 請求異常: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
            
            response_data = response.json()
            
            # 提取響應文本
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                if 'content' in response_data['candidates'][0]:
                    if 'parts' in response_data['candidates'][0]['content']:
                        response_text = response_data['candidates'][0]['content']['parts'][0].get('text', '').strip()
                    else:
                        print("⚠️ 響應格式不正確：缺少 parts")
                        return None
                else:
                    print("⚠️ 響應格式不正確：缺少 content")
                    return None
            else:
                print("⚠️ 響應格式不正確：缺少 candidates")
                return None
            
            # 移除可能的 markdown 代碼塊標記
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # 解析 JSON
            gemini_analysis = json.loads(response_text)
            return gemini_analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Gemini 返回的 JSON 解析失敗: {e}")
            print(f"原始響應前500字符: {response_text[:500]}")
            return None
        except Exception as e:
            print(f"⚠️ Gemini API 調用失敗: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_pixel_to_cm_ratio(self, image_rgb, gemini_analysis):
        """
        根據 Gemini 的參考物分析計算精確的 pixel_to_cm_ratio
        """
        h, w = image_rgb.shape[:2]
        
        if gemini_analysis and 'reference_analysis' in gemini_analysis:
            ref = gemini_analysis['reference_analysis']
            
            # 優先使用 Gemini 直接估算的 ratio
            if 'pixel_to_cm_ratio_estimate' in ref:
                ratio = ref['pixel_to_cm_ratio_estimate']
                confidence = ref.get('ratio_confidence', 0.5)
                print(f"  - 使用 Gemini 估算的 pixel_to_cm_ratio: {ratio:.4f} (置信度: {confidence:.2f})")
                return ratio
            
            # 或基於檢測到的物體計算
            if 'detected_objects' in ref:
                for obj in ref['detected_objects']:
                    if obj['object'] == '盤子' and 'estimated_diameter_cm' in obj:
                        # 假設盤子佔圖像寬度的 70-90%
                        plate_diameter_cm = obj['estimated_diameter_cm']
                        plate_pixels = w * 0.8  # 假設盤子佔寬度 80%
                        ratio = plate_diameter_cm / plate_pixels
                        print(f"  - 基於盤子直徑計算的 pixel_to_cm_ratio: {ratio:.4f}")
                        return ratio
        
        # 默認值（更保守）
        default_ratio = 0.04
        print(f"  - 使用默認 pixel_to_cm_ratio: {default_ratio:.4f}")
        return default_ratio
    
    def estimate_volume_with_gemini_calibration(self, mask, depth_map, 
                                                gemini_analysis, 
                                                pixel_to_cm_ratio,
                                                food_type):
        """
        使用 Gemini 的校準參數估算體積
        """
        area_pixels = np.sum(mask)
        region_depth = depth_map[mask]
        avg_depth = np.mean(region_depth) if len(region_depth) > 0 else 0.3
        
        # 面積轉換
        area_cm2 = area_pixels * (pixel_to_cm_ratio ** 2)
        
        # 使用 Gemini 的深度分析
        if gemini_analysis and 'depth_analysis' in gemini_analysis:
            depth_info = gemini_analysis['depth_analysis']
            max_depth_cm = depth_info.get('max_depth_cm', 3.0)
            avg_depth_cm = depth_info.get('overall_average_depth_cm', 2.5)
            packing_density = depth_info.get('packing_density', 0.7)
        else:
            max_depth_cm = 3.0
            avg_depth_cm = 2.5
            packing_density = 0.7
        
        # 深度轉換（使用 Gemini 的深度範圍）
        depth_cm = avg_depth * max_depth_cm
        
        # 體積計算
        volume_cm3 = area_cm2 * depth_cm
        
        # 使用 Gemini 的密度和校準因子
        density = 0.4  # 默認
        
        if gemini_analysis:
            # 查找對應成分的密度
            if 'component_analysis' in gemini_analysis:
                for comp in gemini_analysis['component_analysis']:
                    if comp['name'] == food_type:
                        density = comp.get('density_factor', 0.4)
                        break
            
            # 使用整體校準因子
            if 'calibration_factors' in gemini_analysis:
                volume_correction = gemini_analysis['calibration_factors'].get(
                    'overall_volume_correction', 1.0
                )
                volume_cm3 *= volume_correction
        
        # 最終重量計算（考慮密度和空隙）
        weight_grams = volume_cm3 * density * packing_density
        
        return weight_grams
    
    def calculate_nutrition_gemini(self, segments, image_rgb, depth_map,
                                  gemini_analysis, pixel_to_cm_ratio):
        """
        結合 Gemini 分析的營養計算
        """
        component_dict = {}
        
        # 如果有 Gemini 分析，提取成分信息
        gemini_components = {}
        total_gemini_weight = 0
        if gemini_analysis and 'component_analysis' in gemini_analysis:
            for comp in gemini_analysis['component_analysis']:
                gemini_components[comp['name']] = {
                    'weight_g': comp.get('estimated_weight_g', 0),
                    'volume_ratio': comp.get('volume_ratio', 0),
                    'confidence': comp.get('confidence', 0.5)
                }
                total_gemini_weight += comp.get('estimated_weight_g', 0)
        
        # 處理每個 SAM 分割區域
        total_computed_weight = 0
        for segment_mask in segments:
            masked_image = image_rgb.copy()
            masked_image[~segment_mask] = 0
            region_image = masked_image[
                np.any(segment_mask, axis=1)
            ][:, np.any(segment_mask, axis=0)]
            
            if region_image.size == 0:
                continue
            
            # 本地識別
            food_type = self.identify_food_component(region_image)
            
            # 使用 Gemini 校準的體積估算
            weight = self.estimate_volume_with_gemini_calibration(
                segment_mask, depth_map, gemini_analysis,
                pixel_to_cm_ratio, food_type
            )
            total_computed_weight += weight
            
            # 累加同類型成分
            if food_type in component_dict:
                component_dict[food_type]['weight_g'] += weight
            else:
                component_dict[food_type] = {
                    'type': food_type,
                    'weight_g': weight
                }
        
        # 如果有 Gemini 的總重量估算，進行校準
        calibration_applied = False
        if gemini_analysis and 'nutrition_estimate' in gemini_analysis:
            gemini_weight_range = gemini_analysis['nutrition_estimate'].get(
                'total_weight_g_range', None
            )
            if gemini_weight_range and total_computed_weight > 0:
                # 使用範圍的中值
                gemini_weight = (gemini_weight_range[0] + gemini_weight_range[1]) / 2
                calibration_factor = gemini_weight / total_computed_weight
                
                # 限制校準係數範圍（避免過度調整）
                calibration_factor = np.clip(calibration_factor, 0.3, 2.0)
                
                print(f"  - Gemini 校準係數: {calibration_factor:.2f}")
                print(f"    (計算重量: {total_computed_weight:.1f}g, Gemini估算: {gemini_weight:.1f}g)")
                
                for food_type in component_dict:
                    component_dict[food_type]['weight_g'] *= calibration_factor
                calibration_applied = True
        
        # 計算營養數據
        component_details = []
        total_nutrition = {
            'calories': 0, 'protein': 0, 'carbs': 0,
            'fat': 0, 'fiber': 0, 'weight': 0
        }
        
        for food_type, data in component_dict.items():
            weight = data['weight_g']
            nutrition_per_100g = NUTRITION_DB.get(food_type, NUTRITION_DB['未知成分'])
            
            comp = {
                'type': food_type,
                'weight_g': weight,
                'calories': weight * nutrition_per_100g['calories'] / 100,
                'protein': weight * nutrition_per_100g['protein'] / 100,
                'carbs': weight * nutrition_per_100g['carbs'] / 100,
                'fat': weight * nutrition_per_100g['fat'] / 100,
                'fiber': weight * nutrition_per_100g['fiber'] / 100,
            }
            component_details.append(comp)
            
            for key in total_nutrition:
                if key == 'weight':
                    total_nutrition[key] += comp['weight_g']
                else:
                    total_nutrition[key] += comp[key]
        
        component_details.sort(key=lambda x: x['weight_g'], reverse=True)
        
        # 保存 Gemini 分析結果供後續使用
        self.gemini_analysis = gemini_analysis
        
        return total_nutrition, component_details
    
    def analyze_salad_with_gemini(self, image_path, visualize=True):
        """
        使用 Gemini + SAM + DPT 的完整分析流程
        """
        import time
        
        total_start_time = time.time()
        
        print("\n" + "="*60)
        print("混合分析模式：Gemini Vision + SAM + DPT")
        print("="*60)
        
        # 步驟 1: Gemini 分析
        gemini_analysis = None
        if self.use_gemini:
            print("\n[步驟 1/5] 使用 Gemini Vision 進行智能分析...")
            step_start = time.time()
            gemini_analysis = self.analyze_with_gemini(image_path)
            step_time = time.time() - step_start
            
            if gemini_analysis:
                print("✓ Gemini 分析完成")
                print(f"  耗時: {step_time:.2f} 秒")
                print(f"  - 識別到 {len(gemini_analysis.get('component_analysis', []))} 種成分")
                if 'nutrition_estimate' in gemini_analysis:
                    cal_range = gemini_analysis['nutrition_estimate']['total_calories_range']
                    print(f"  - 估算熱量範圍: {cal_range[0]}-{cal_range[1]} 大卡")
            else:
                print("⚠️ Gemini 分析失敗，將使用默認參數")
        
        # 步驟 2: 載入圖像
        print("\n[步驟 2/5] 載入原始圖像...")
        step_start = time.time()
        image_rgb = self.load_image(image_path)
        step_time = time.time() - step_start
        print(f"  耗時: {step_time:.2f} 秒")
        
        # 計算 pixel_to_cm_ratio
        pixel_to_cm_ratio = self.calculate_pixel_to_cm_ratio(
            image_rgb, gemini_analysis
        )
        
        # 步驟 3: SAM 分割
        print("\n[步驟 3/5] 使用 SAM 進行精確分割...")
        step_start = time.time()
        segments = self.segment_image_with_sam(image_rgb)
        step_time = time.time() - step_start
        print(f"✓ SAM 識別到 {len(segments)} 個區域")
        print(f"  耗時: {step_time:.2f} 秒")
        
        # 步驟 4: DPT 深度估計
        print("\n[步驟 4/5] 使用 DPT 估計深度...")
        step_start = time.time()
        depth_map = self.estimate_depth_with_dpt(image_rgb)
        step_time = time.time() - step_start
        print("✓ 深度圖已生成")
        print(f"  耗時: {step_time:.2f} 秒")
        
        # 步驟 5: 結合所有信息計算營養
        print("\n[步驟 5/5] 結合 Gemini 分析計算營養數據...")
        step_start = time.time()
        total_nutrition, component_details = self.calculate_nutrition_gemini(
            segments, image_rgb, depth_map, gemini_analysis, pixel_to_cm_ratio
        )
        step_time = time.time() - step_start
        print("✓ 營養數據計算完成")
        print(f"  耗時: {step_time:.2f} 秒")
        
        # 可視化
        if visualize:
            print("\n正在生成最終可視化結果...")
            step_start = time.time()
            self.visualize_results_with_chinese(image_rgb, segments, depth_map, 
                                               total_nutrition, component_details, gemini_analysis)
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
        
        return total_nutrition, component_details, gemini_analysis
    
    def visualize_results_with_chinese(self, image_rgb, segments, depth_map, 
                                       total_nutrition, component_details, gemini_analysis=None):
        """
        可視化結果（支持中文顯示）
        """
        import matplotlib.pyplot as plt
        import matplotlib
        import os
        import cv2
        
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
        
        # 如果有 Gemini 分析，添加備註
        if gemini_analysis and 'nutrition_estimate' in gemini_analysis:
            cal_range = gemini_analysis['nutrition_estimate'].get('total_calories_range', [])
            if cal_range:
                nutrition_text += f"\n\n(Gemini 估算範圍: {cal_range[0]}-{cal_range[1]} 大卡)"
        
        # 使用支持中文的字體顯示文本
        axes[1, 1].text(0.05, 0.95, nutrition_text, fontsize=11, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        family=plt.rcParams['font.sans-serif'][0] if font_set else 'sans-serif')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 為標題留出空間
        result_path = os.path.join(self.result_dir, '05_final_analysis_result.png')
        plt.savefig(result_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 最終結果已保存到: {result_path}")
        plt.close()


def main():
    """主函數"""
    print("=" * 60)
    print("沙拉營養分析系統（Gemini 增強版）")
    print("=" * 60)
    
    # 初始化分析器（會自動使用 Gemini API）
    # API key 從環境變量 GEMINI_API_KEY 讀取，或在此處傳入
    import os
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    analyzer = GeminiSaladAnalyzer(
        sam_checkpoint_path=None,
        use_gemini=True,
        gemini_api_key=gemini_api_key  # 從環境變量讀取，或傳入 None 使用環境變量
    )
    
    # 示例：分析圖像
    image_path = input("請輸入沙拉圖像路徑（或按Enter使用測試模式）: ").strip()
    
    if not image_path:
        print("未提供圖像路徑，請在代碼中指定圖像路徑")
        return
    
    # 執行分析
    try:
        total_nutrition, component_details, gemini_info = analyzer.analyze_salad_with_gemini(image_path)
        
        # 打印結果
        print("\n" + "=" * 60)
        print("營養分析結果")
        print("=" * 60)
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
        
        # 如果有 Gemini 分析結果，顯示詳細信息
        if gemini_info:
            print("\n" + "=" * 60)
            print("Gemini 分析詳情")
            print("=" * 60)
            print(json.dumps(gemini_info, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

