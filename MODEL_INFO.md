# SAM 和 DPT 模型信息

## 1. SAM (Segment Anything Model)

### 模型類型
- **模型版本**: SAM ViT-H (Vision Transformer Huge)
- **檢查點文件**: `sam_vit_h_4b8939.pth`
- **模型架構**: Vision Transformer (ViT) - Huge 版本
- **參數規模**: 約 2.4GB

### 模型初始化參數
```python
sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
```

### SamAutomaticMaskGenerator 參數
當前使用的自動mask生成器參數如下：

| 參數名稱 | 值 | 說明 |
|---------|-----|------|
| `points_per_side` | 32 | 每個邊的採樣點數（總共 32×32 = 1024 個點） |
| `pred_iou_thresh` | 0.86 | 預測IoU閾值，過濾低質量mask |
| `stability_score_thresh` | 0.92 | 穩定性分數閾值，確保mask穩定 |
| `crop_n_layers` | 1 | 裁剪層數，用於處理大圖像 |
| `crop_n_points_downscale_factor` | 2 | 裁剪時點的下採樣因子 |
| `min_mask_region_area` | 100 | 最小mask區域面積（像素），過濾小區域 |

### 模型下載地址
```
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### SAM 模型特點
- **高精度**: ViT-H 是SAM最大的模型，分割精度最高
- **無需預訓練**: 可以分割任何物體，無需特定類別訓練
- **自動生成**: 使用自動mask生成器可以自動識別圖像中的所有物體

---

## 2. DPT (Dense Prediction Transformer) - MiDaS

### 模型類型
- **默認模型**: DPT-Hybrid
- **模型架構**: DPT-Hybrid (ViT-B + ResNet-50 backbone)
- **輸入尺寸**: 384×384 像素
- **來源**: Intel ISL MiDaS 項目

### 當前使用的模型
```python
dpt_model_type='dpt_hybrid'  # 默認值
DPT_Hybrid(pretrained=True)
```

### 可選的DPT模型類型
系統支持以下DPT模型類型：

| 模型類型 | 函數名稱 | Backbone | 輸入尺寸 | 說明 |
|---------|---------|----------|---------|------|
| `dpt_hybrid` | `DPT_Hybrid` | ViT-B + ResNet-50 | 384×384 | 默認，平衡精度和速度 |
| `dpt_large` | `DPT_Large` | ViT-L | 384×384 | 更高精度，較慢 |
| `dpt_beit_large_384` | `DPT_BEiT_L_384` | BEiT-L | 384×384 | BEiT架構，高精度 |
| `dpt_beit_base_384` | `DPT_BEiT_B_384` | BEiT-B | 384×384 | BEiT架構，基礎版 |

### DPT 深度估計參數

#### 輸入處理
```python
# 圖像調整
img_tensor = F.interpolate(
    img_tensor, 
    size=(384, 384),  # 調整到模型輸入尺寸
    mode='bilinear', 
    align_corners=False
)
```

#### 深度預測
```python
# 推理模式
with torch.no_grad():
    depth = self.dpt_model(img_tensor)
    # 調整回原始圖像尺寸
    depth = F.interpolate(
        depth.unsqueeze(1),
        size=image_rgb.shape[:2],  # 原始圖像尺寸
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
```

#### 深度圖正規化
```python
# 正規化到 0-1 範圍
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
```

### 模型載入方式
系統會優先從本地MiDaS資料夾載入模型：
```
D:\AI_CODE\isl-org-MiDaS-4545977
```

如果本地找不到，會自動從網路下載：
```python
torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
```

### DPT-Hybrid 模型特點
- **混合架構**: 結合 Vision Transformer (ViT-B) 和 ResNet-50
- **高精度**: 在單目深度估計任務上表現優異
- **通用性**: 無需特定場景訓練即可工作
- **速度**: 相比DPT-Large更快，適合實時應用

---

## 3. 使用示例

### 更改SAM模型參數
```python
analyzer = SaladNutritionAnalyzer()

# 如果想調整自動mask生成器參數，可以在初始化後修改：
analyzer.sam_mask_generator = SamAutomaticMaskGenerator(
    analyzer.sam_model,
    points_per_side=64,  # 增加採樣點密度（更慢但更精確）
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=200,  # 過濾更小的區域
)
```

### 更改DPT模型類型
```python
# 使用更大的DPT-Large模型（更慢但更精確）
analyzer = SaladNutritionAnalyzer(dpt_model_type='dpt_large')

# 或使用BEiT架構的模型
analyzer = SaladNutritionAnalyzer(dpt_model_type='dpt_beit_large_384')
```

---

## 4. 性能考慮

### SAM ViT-H
- **內存需求**: 約 6-8GB GPU內存
- **推理速度**: 中等（自動mask生成較慢）
- **精度**: 最高

### DPT-Hybrid
- **內存需求**: 約 2-3GB GPU內存
- **推理速度**: 較快（384×384輸入）
- **精度**: 高（適合大多數應用）

### 優化建議
- 如果GPU內存不足，可以考慮使用SAM ViT-B或ViT-L（較小的版本）
- 如果需要更快的分割速度，可以減少`points_per_side`參數
- 如果需要更精確的深度估計，可以使用DPT-Large或DPT-BEiT模型

