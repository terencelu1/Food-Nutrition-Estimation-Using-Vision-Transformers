"""
使用範例：沙拉營養分析
"""

from salad_nutrition_analyzer import SaladNutritionAnalyzer
import os

def example_usage():
    """使用範例"""
    
    print("=" * 60)
    print("沙拉營養分析系統 - 使用範例")
    print("=" * 60)
    
    # 檢查SAM模型是否存在
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        print(f"\n警告: 未找到SAM模型文件 '{sam_checkpoint}'")
        print("請從以下網址下載:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print(f"\n將文件放在當前目錄: {os.getcwd()}")
        print("\n將使用簡化版分割功能繼續...\n")
        sam_checkpoint = None
    
    # 初始化分析器（結果會自動保存到result資料夾）
    print("\n正在初始化分析器...")
    analyzer = SaladNutritionAnalyzer(sam_checkpoint_path=sam_checkpoint, result_dir='result')
    
    # 獲取圖像路徑
    print("\n請輸入沙拉圖像的路徑:")
    image_path = input("圖像路徑: ").strip().strip('"').strip("'")
    
    if not image_path or not os.path.exists(image_path):
        print(f"\n錯誤: 找不到圖像文件 '{image_path}'")
        print("\n請確保:")
        print("1. 圖像路徑正確")
        print("2. 使用絕對路徑或相對路徑")
        print("3. 圖像文件存在且可讀取")
        return
    
    # 執行分析
    try:
        print(f"\n正在分析圖像: {image_path}")
        total_nutrition, component_details = analyzer.analyze_salad(
            image_path, 
            visualize=True
        )
        
        # 顯示結果
        print("\n" + "=" * 60)
        print("營養分析結果")
        print("=" * 60)
        print(f"\n📊 總體數據:")
        print(f"  總重量: {total_nutrition['weight']:.1f} 克")
        print(f"  總熱量: {total_nutrition['calories']:.1f} 大卡")
        print(f"  蛋白質: {total_nutrition['protein']:.1f} 克")
        print(f"  碳水化合物: {total_nutrition['carbs']:.1f} 克")
        print(f"  脂肪: {total_nutrition['fat']:.1f} 克")
        print(f"  纖維: {total_nutrition['fiber']:.1f} 克")
        
        print(f"\n🍽️  成分詳情:")
        for i, comp in enumerate(component_details, 1):
            print(f"  {i}. {comp['type']}")
            print(f"     重量: {comp['weight_g']:.1f} 克")
            print(f"     熱量: {comp['calories']:.1f} 大卡")
            print(f"     蛋白質: {comp['protein']:.1f} 克")
        
        print("\n✅ 分析完成！")
        print(f"所有處理結果已保存到 '{analyzer.result_dir}' 資料夾:")
        print("  - 01_original_image.jpg (原始圖像)")
        print("  - 02_sam_all_masks.jpg (SAM所有masks)")
        print("  - 03_sam_segmentation_result.jpg (SAM分割結果)")
        print("  - 04_dpt_depth_map.jpg (DPT深度圖)")
        print("  - 05_final_analysis_result.png (最終分析結果)")
        
    except Exception as e:
        print(f"\n❌ 分析過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        print("\n請檢查:")
        print("1. 圖像格式是否正確（支持JPG、PNG等）")
        print("2. 所有依賴套件是否已安裝")
        print("3. 是否有足夠的內存/GPU內存")

if __name__ == "__main__":
    example_usage()

