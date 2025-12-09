# 沙拉營養分析系統

使用SAM (Segment Anything Model) 和 DPT-Hybrid (MiDaS) 進行圖像分析和營養計算。

## 安裝步驟

1. 安裝Python套件：
```bash
# 按照environment.txt中的內容逐行執行，或使用以下命令批量安裝
pip install torch torchvision torchaudio
pip install transformers
pip install segment-anything
pip install opencv-python
pip install numpy
pip install Pillow
pip install matplotlib
pip install scikit-image
pip install scikit-learn
pip install timm
pip install opencv-contrib-python
pip install gradio
```

2. 下載SAM模型檢查點：
   - 訪問：https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   - 將下載的文件放在與 `salad_nutrition_analyzer.py` 相同的目錄中
   - 或修改代碼中的 `sam_checkpoint_path` 參數指向下載的文件

## 使用方法

### 基本使用

```python
from salad_nutrition_analyzer import SaladNutritionAnalyzer

# 初始化分析器
analyzer = SaladNutritionAnalyzer(sam_checkpoint_path="sam_vit_h_4b8939.pth")

# 分析沙拉圖像
total_nutrition, component_details = analyzer.analyze_salad("path/to/salad_image.jpg")

# 查看結果
print(f"總熱量: {total_nutrition['calories']:.1f} 大卡")
print(f"總重量: {total_nutrition['weight']:.1f} 克")
```

### 直接運行

```bash
python salad_nutrition_analyzer.py
```

運行後會提示輸入圖像路徑。

## 功能說明

### 1. 圖像分割（SAM）
- 自動識別和分割沙拉中的不同食物成分
- 如果SAM模型不可用，會使用簡化版的K-means聚類分割

### 2. 深度估計（DPT-Hybrid）
- 使用MiDaS DPT-Hybrid模型估計圖像深度
- 用於估算食物的體積和重量

### 3. 營養計算
- 自動識別食物類型（基於顏色特徵）
- 計算每個成分的重量
- 輸出總營養數據（熱量、蛋白質、碳水化合物、脂肪、纖維）

## 營養數據庫

系統內建常見沙拉食材的營養數據（每100克）：
- 生菜、番茄、黃瓜、胡蘿蔔、玉米
- 雞蛋、雞胸肉、堅果、起司、橄欖
- 未知成分（默認值）

## 輸出結果

分析完成後會：
1. 顯示可視化結果圖（原始圖像、分割結果、深度圖、營養數據）
2. 保存結果圖到 `salad_analysis_result.png`
3. 在控制台輸出詳細的營養數據

## 注意事項

1. **體積估算精度**：深度估計和體積計算的精度取決於：
   - 圖像質量
   - 拍攝角度
   - `pixel_to_cm_ratio` 參數（需根據實際情況調整）

2. **食物識別**：當前版本使用基於顏色的簡單分類。要獲得更高精度，建議：
   - 訓練專用的食物分類模型
   - 使用更先進的圖像識別API（如Google Vision API）

3. **模型下載**：
   - SAM模型文件較大（約2.4GB），需要足夠的磁盤空間
   - DPT模型會在使用時自動從torch.hub下載

## 故障排除

- 如果SAM模型載入失敗，系統會自動使用簡化版分割
- 如果DPT模型載入失敗，會使用默認深度圖
- 確保有足夠的GPU內存（如果使用GPU）或使用CPU模式

