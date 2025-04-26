import logging
from pathlib import Path
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class OCRPipeline:
    """
    In-memory conversion of images in a directory to PDF buffers.
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
    # Example usage; adjust "your/image/dir" as needed
    pipeline = OCRPipeline(image_dir="your/image/dir")
    pipeline.convert_images_to_pdf()
    # Now pipeline._pdf_buffers holds all PDFs in memory
