# train_yolov11.py
import os
from ultralytics import YOLO

def main():
    # 1) Paths and hyperparameters
    project_dir = os.path.abspath(os.path.dirname(__file__))
    data_yaml   = os.path.join(project_dir, 'data.yaml')
    model_name  = 'yolov11l.pt'    # or your custom YOLOv11 large weights path
    epochs      = 50
    imgsz       = (1200, 800)      # width, height
    batch       = 8                # reduce batch size for high-res
    device      = '0'              # GPU index or 'cpu'

    # 2) Sanity checks
    for split in ('train', 'val', 'test'):
        img_folder = os.path.join(project_dir, split, 'images')
        lbl_folder = os.path.join(project_dir, split, 'labels')
        if not os.path.isdir(img_folder):
            raise FileNotFoundError(f"Missing folder: {img_folder}")
        if not os.path.isdir(lbl_folder):
            raise FileNotFoundError(f"Missing folder: {lbl_folder}")

    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Missing data file: {data_yaml}")

    # 3) Load model and start training with augmentation
    model = YOLO(model_name)

    # built-in augmentation params (see https://docs.ultralytics.com)
    aug_kwargs = dict(
        # enable basic augment pipelines
        mosaic          = True,      # mix 4 images
        mixup           = 0.15,      # 15% mixup
        hsv_h           = 0.015,     # hue augmentation (±1.5%)
        hsv_s           = 0.7,       # saturation augmentation (±70%)
        hsv_v           = 0.4,       # value augmentation (±40%)
        degrees         = 2.0,       # rotation ±2°
        translate       = 0.08,      # translate ±8%
        scale           = 0.5,       # scale ±50%
        shear           = 0.0,       # shear ±0°
        perspective     = 0.0,       # perspective ±0%
        flipud          = 0.0,       # vertical flip probability
        fliplr          = 0.5,       # horizontal flip probability
        grayscale       = 0.15       # convert to grayscale with 15% chance
    )

    results = model.train(
        data        = data_yaml,
        epochs      = epochs,
        imgsz       = imgsz,
        batch       = batch,
        device      = device,
        project     = project_dir,
        name        = 'runs/train/yolov11l_gray15',
        exist_ok    = False,     # error if run exists
        save        = True,
        save_period = -1,        # save only best + final
        augment     = True,      # enable augmentation
        augment_kwargs = aug_kwargs
    )

    # 4) Report
    best_pt = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"\n✅ Training complete!\n▶️ Best model: {best_pt}")

if __name__ == '__main__':
    main()
