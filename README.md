# 沙拉營養分析系統

使用 SAM (Segment Anything Model) + DPT-Hybrid (MiDaS) + Gemini Vision API 進行智能沙拉營養分析。

## 功能特點

- **圖像分割**：使用 SAM 模型精確分割沙拉中的各個成分
- **深度估計**：使用 DPT-Hybrid 模型估算食物體積
- **智能分析**：整合 Gemini Vision API 進行更精準的參數校準和營養估算
- **營養計算**：自動計算總熱量、蛋白質、碳水化合物、脂肪、纖維等營養成分
- **中文報告**：生成包含可視化結果的中文營養分析報告

## 安裝要求

### 環境設置

```bash
# 使用 conda 環境（推薦）
conda activate yolo_test

# 安裝依賴
pip install torch torchvision
pip install timm
pip install segment-anything
pip install opencv-python
pip install numpy
pip install matplotlib
pip install requests
```

### 模型文件

下載 SAM 模型文件並放置在項目根目錄：
- `sam_vit_h_4b8939.pth` (SAM ViT-H 模型)

### API Key 設置

**重要**：請設置環境變量 `GEMINI_API_KEY`，不要將 API key 硬編碼在代碼中。

```bash
# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# Windows CMD
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

或在代碼中傳入：

```python
analyzer = GeminiSaladAnalyzer(
    sam_checkpoint_path="sam_vit_h_4b8939.pth",
    use_gemini=True,
    gemini_api_key="your_api_key_here"  # 可選，優先使用環境變量
)
```

## 使用方法

### 基礎版本（不使用 Gemini）

```python
from salad_nutrition_analyzer import SaladNutritionAnalyzer

analyzer = SaladNutritionAnalyzer(
    sam_checkpoint_path="sam_vit_h_4b8939.pth"
)

total_nutrition, component_details = analyzer.analyze_salad("path/to/image.jpg")
```

### Gemini 增強版（推薦）

```python
from salad_nutrition_analyzer_gemini import GeminiSaladAnalyzer
import os

analyzer = GeminiSaladAnalyzer(
    sam_checkpoint_path="sam_vit_h_4b8939.pth",
    use_gemini=True,
    gemini_api_key=os.getenv('GEMINI_API_KEY')
)

total_nutrition, component_details, gemini_info = analyzer.analyze_salad_with_gemini("path/to/image.jpg")
```

## 項目結構

```
.
├── salad_nutrition_analyzer.py          # 基礎版本（SAM + DPT）
├── salad_nutrition_analyzer_gemini.py   # Gemini 增強版
├── README.md                            # 本文件
├── README_salad_analyzer.md             # 詳細使用說明
├── MODEL_INFO.md                        # 模型信息
├── .gitignore                           # Git 忽略文件
└── result/                              # 結果輸出目錄
    ├── 01_original_image.jpg
    ├── 02_sam_all_masks.jpg
    ├── 03_sam_segmentation_result.jpg
    ├── 04_dpt_depth_map.jpg
    └── 05_final_analysis_result.png
```

## 注意事項

- 模型文件（`.pth`）較大，已添加到 `.gitignore`，不會上傳到 Git
- 結果圖片和敏感信息（如 API key）不會被提交到版本控制
- 確保有足夠的 GPU 內存運行 SAM 和 DPT 模型
- Gemini API 需要有效的 API key 和網絡連接

## 授權

請參考各子模塊的授權信息。

