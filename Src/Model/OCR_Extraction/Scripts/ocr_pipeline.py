import logging
from pathlib import Path
from io import BytesIO
import argparse
from PIL import Image
from pdf2image import convert_from_bytes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class OCRPipeline:
    """
    In-memory conversion of images to PDF buffers, rendering to images, and cropping into regions.
    """
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self._pdf_buffers: dict[str, BytesIO] = {}
        self._page_images: dict[str, Image.Image] = {}
        self._regions: dict[str, dict[str, Image.Image]] = {}

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

    def render_pdfs_to_images(self):
        """Render each in-memory PDF buffer back to a PIL image."""
        total = len(self._pdf_buffers)
        logging.info(f"[STEP 2] Rendering {total} PDFs to images…")
        for idx, (name, pdf_buf) in enumerate(self._pdf_buffers.items(), start=1):
            logging.info(f"  • ({idx}/{total}) Rendering '{name}'")
            try:
                pages = convert_from_bytes(pdf_buf.getvalue(), dpi=300)
                if pages:
                    self._page_images[name] = pages[0]
                    logging.info(f"    Rendered image for {name}")
            except Exception as e:
                logging.warning(f"Failed to render {name}: {e}")
        logging.info(f"Rendered {len(self._page_images)} images successfully.")

    def crop_regions(self):
        """Crop each rendered page image into three table regions: Symbol, Value1, Value2."""
        total = len(self._page_images)
        logging.info(f"[STEP 3] Cropping {total} images into regions…")
        for idx, (name, img) in enumerate(self._page_images.items(), start=1):
            logging.info(f"  • ({idx}/{total}) Cropping '{name}'")
            w, h = img.size
            bounds = [
                (0,            int(w * 0.12)),  # Symbol region
                (int(w * 0.12), int(w * 0.52)), # Value1 region
                (int(w * 0.52), w)              # Value2 region
            ]
            region_imgs = {}
            for i, label in enumerate(['Symbol', 'Value1', 'Value2']):
                left, right = bounds[i]
                region_imgs[label] = img.crop((left, 0, right, h))
                logging.info(f"    Cropped {label} for {name}")
            self._regions[name] = region_imgs
        logging.info(f"Cropped regions for {len(self._regions)} images.")

    def extract_table(self):
        """
        Placeholder for OCR logic on cropped regions.
        """
        # TODO: apply OCR to self._regions and collect structured data
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert images to PDFs, render, and crop into regions"
    )
    parser.add_argument(
        "--image_dir",
        default="Src/OCR_Extraction_Model/Dataset",
        help="Directory containing image files"
    )
    args = parser.parse_args()

    pipeline = OCRPipeline(image_dir=args.image_dir)
    pipeline.convert_images_to_pdf()
    pipeline.render_pdfs_to_images()
    pipeline.crop_regions()
    # Next: pipeline.extract_table() for OCR
