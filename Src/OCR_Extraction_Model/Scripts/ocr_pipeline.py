import logging
from pathlib import Path
from io import BytesIO
import argparse
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class OCRPipeline:
    """
    In‐memory conversion of images to PDF buffers.
    """
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self._pdf_buffers: dict[str, BytesIO] = {}

    def convert_images_to_pdf(self):
        """Convert images in the directory into in-memory PDF buffers."""
        logging.info(f"Looking for images in: {self.image_dir.resolve()}")
        imgs = [
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        ]
        logging.info(f"[STEP 1] Converting {len(imgs)} images to PDF buffers…")
        for img_path in imgs:
            logging.info(f"  • {img_path.name}")
            try:
                img = Image.open(img_path).convert('RGB')
                buf = BytesIO()
                img.save(buf, format='PDF')
                buf.seek(0)
                self._pdf_buffers[img_path.stem] = buf
            except Exception as e:
                logging.warning(f"Failed to convert {img_path.name}: {e}")
        logging.info(f"Converted {len(self._pdf_buffers)} images successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to in-memory PDFs")
    parser.add_argument(
        "--image_dir",
        default="Src/OCR_Extraction_Model/Dataset",
        help="Directory containing image files"
    )
    args = parser.parse_args()

    pipeline = OCRPipeline(image_dir=args.image_dir)
    pipeline.convert_images_to_pdf()
    # Access your PDFs via pipeline._pdf_buffers
