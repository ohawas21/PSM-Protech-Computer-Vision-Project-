import cv2
import os
import random

# === CONFIGURATION ===
input_folder = r"C:\Users\Admin\psm\PSM-Protech-Feasibility-Study\Src\Dataset\OCR_Classification_Data"
output_folder = r"C:\Users\Admin\psm\PSM-Protech-Feasibility-Study\Src\Dataset\final_cs"
num_copies = 20
class_id = 0  # YOLO class ID for Box 1

# === SETUP ===
os.makedirs(output_folder, exist_ok=True)

# === PROCESS EACH IMAGE ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        box_width = w // 3
        boxes = [
            (0, 0, box_width, h),               # Box 1
            (box_width, 0, 2 * box_width, h),   # Box 2
            (2 * box_width, 0, w, h)            # Box 3
        ]

        name, ext = os.path.splitext(filename)

        for i in range(1, num_copies + 1):
            img_copy = img.copy()

            # === Draw dividing black lines ===
            cv2.line(img_copy, (box_width, 0), (box_width, h), (0, 0, 0), 2)
            cv2.line(img_copy, (2 * box_width, 0), (2 * box_width, h), (0, 0, 0), 2)

            for j, (x1, y1, x2, y2) in enumerate(boxes):
                if j in [1, 2]:  # Add random number in Box 2 and 3
                    # Either 1-digit float or 2-digit int
                    if random.random() < 0.5:
                        rand_num = f"{random.uniform(1.0, 9.9):.1f}"
                    else:
                        rand_num = str(random.randint(10, 99))

                    cv2.putText(
                        img_copy,
                        rand_num,
                        (x1 + 10, y1 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),  # Blue text
                        2
                    )

            # === Save Image ===
            out_img_name = f"{name}_annotated_{i}{ext}"
            out_img_path = os.path.join(output_folder, out_img_name)
            cv2.imwrite(out_img_path, img_copy)

            # === Save YOLO Annotation (Box 1 only) ===
            x1, y1, x2, y2 = boxes[0]
            box_w = x2 - x1
            box_h = y2 - y1
            x_center = (x1 + box_w / 2) / w
            y_center = (y1 + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
            txt_path = os.path.join(output_folder, f"{name}_annotated_{i}.txt")
            with open(txt_path, 'w') as f:
                f.write(yolo_line)

print("✅ Complete: Divided into 3 with black lines, numbers added, YOLO Box 1 only.")
