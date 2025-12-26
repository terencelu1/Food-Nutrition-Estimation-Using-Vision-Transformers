"""
YOLOv8 Fast Training Script - 10 Minutes Version
Optimized for RTX 4050 6GB VRAM
Purpose: Quick training to generate training charts
"""

from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import time

def create_dataset_distribution_chart():
    """Create dataset distribution charts"""
    labels = ['Train', 'Validation', 'Test']
    sizes = [6721, 1500, 180]
    colors = ['#66b3ff', '#99ff99', '#ffcc99']
    explode = (0.05, 0.05, 0.1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
            startangle=90, explode=explode, shadow=True)
    ax1.set_title('Dataset Distribution (Train/Val/Test)', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = ax2.bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Dataset Size Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{size:,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    print("Dataset distribution chart saved: dataset_distribution.png")
    plt.close()

def train_yolo_fast():
    """Fast training YOLOv8 model (10 minutes version)"""
    print("=" * 60)
    print("Fast Training YOLOv8 Model (10 Minutes Version)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Load smallest pretrained model
    print("\nLoading pretrained model: yolov8n.pt")
    model = YOLO('yolov8n.pt')
    
    # Fast training configuration
    print("\nFast Training Configuration:")
    print("  - Model: YOLOv8n (nano)")
    print("  - Epochs: 8 (for quick demonstration)")
    print("  - Image size: 320 (reduced computation)")
    print("  - Batch size: 32 (maximize GPU utilization)")
    print("  - Validation frequency: Every 2 epochs")
    print("  - Target time: ~10 minutes")
    
    # Dataset path (use corrected version)
    # Change to dataset directory for training
    dataset_dir = 'Fruits-And-Vegetables-Detection-Dataset-main/LVIS_Fruits_And_Vegetables'
    data_yaml = os.path.join(dataset_dir, 'data_train.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"\nError: Dataset config file not found: {data_yaml}")
        print("Please check the path and try again.")
        return None, 0
    
    # Change to dataset directory for correct relative path resolution
    original_dir = os.getcwd()
    dataset_abs_path = os.path.abspath(dataset_dir)
    
    # Change working directory to dataset directory
    os.chdir(dataset_abs_path)
    data_yaml = 'data_train.yaml'
    
    print(f"\nDataset config: {data_yaml}")
    print(f"Working directory: {os.getcwd()}")
    
    # Start fast training
    # Note: YOLO resolves paths relative to the yaml file location
    # So we change to dataset directory first
    results = model.train(
        data=data_yaml,
        
        # === Core optimization parameters ===
        epochs=16,                  # Very few epochs (for chart demonstration only)
        imgsz=480,                 # Small image size (320 is 4x faster than 640)
        batch=32,                   # Large batch (reduce to 24 or 16 if OOM)
        device=0,                   # GPU
        
        # === Training speed optimization ===
        patience=5,                 # Early stopping (unlikely to trigger with few epochs)
        save=True,                  # Save checkpoints
        plots=True,                 # Generate charts (required!)
        val=True,                   # Validation
        
        # === Project settings ===
        # Save results relative to original directory (will be converted to absolute path)
        project=os.path.join('..', 'yolo_training'),  # Go up one level from dataset dir
        name='fruits_veggies_fast',
        exist_ok=True,
        
        # === Optimizer settings (faster convergence) ===
        optimizer='AdamW',          # AdamW usually converges faster
        lr0=0.02,                   # Slightly higher initial learning rate
        lrf=0.1,                    # Final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,            # Reduce warmup (1 epoch is enough)
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # === Loss weights ===
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # === Performance optimization ===
        workers=4,                  # Data loading processes (adjust based on CPU cores)
        cache=False,                # Don't cache images (save memory, faster startup)
        amp=True,                   # Mixed precision training (faster training speed)
        
        # === Reduce output ===
        verbose=True,
        close_mosaic=0,              # Don't close mosaic (keep data augmentation)
    )
    
    elapsed_time = time.time() - start_time
    
    # Restore original working directory
    os.chdir(original_dir)
    
    print("\n" + "=" * 60)
    print("Fast Training Completed!")
    print("=" * 60)
    print(f"\nActual training time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")
    
    # Results path (relative to original directory or absolute)
    results_path = results.save_dir
    if not os.path.isabs(results_path):
        # If relative path, it might be relative to dataset directory, convert to absolute
        results_path = os.path.join(dataset_abs_path, results_path)
    
    print(f"Best model: {os.path.join(results_path, 'weights', 'best.pt')}")
    print(f"Training charts: {results_path}/")
    
    print("\nAvailable chart files:")
    print("  - results.png (training curves overview)")
    print("  - confusion_matrix.png (confusion matrix)")
    print("  - PR_curve.png (PR curve)")
    print("  - F1_curve.png (F1 curve)")
    print("  - val_batch*_pred.jpg (validation set predictions)")
    
    return results, elapsed_time

if __name__ == "__main__":
    # 1. Create dataset distribution chart
    print("\n[Step 1] Generating dataset distribution chart...")
    create_dataset_distribution_chart()
    
    # 2. Fast training
    print("\n[Step 2] Starting fast training (target: 10 minutes)...")
    print("Note: This configuration is for quick chart generation only, model performance may be low")
    print()
    
    results, train_time = train_yolo_fast()
    
    if results is None:
        print("\nTraining failed. Please check the error messages above.")
        exit(1)
    
    print("\n" + "=" * 60)
    print("All charts generated successfully!")
    print("=" * 60)
    
    if train_time > 600:  # Over 10 minutes
        print(f"\nWarning: Training time ({train_time/60:.2f} minutes) exceeded target of 10 minutes")
        print("Suggestions:")
        print("  - Reduce epochs to 5-6")
        print("  - Reduce imgsz to 256")
        print("  - Increase batch to 48 (if VRAM allows)")
    else:
        print(f"\nTraining time ({train_time/60:.2f} minutes) meets target!")

