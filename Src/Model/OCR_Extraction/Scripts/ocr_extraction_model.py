import re
import logging
from pathlib import Path

import camelot
import pandas as pd

# ─── Configuration ──────────────────────────────────────────────────────────────
PDF_DIR    = Path("path/to/your/native_pdfs")      # ← point at real PDFs, not images
CSV_DIR    = Path("path/to/output_folder")
OUTPUT_CSV = CSV_DIR / "camelot_only_extracted.csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)
# ────────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, 
                    format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def extract_from_pdf(pdf_path: Path):
    # try both stream (text-based) and lattice (line-based)
    for flavor in ("stream", "lattice"):
        try:
            tables = camelot.read_pdf(str(pdf_path), flavor=flavor, pages="1")
            if not tables:
                continue
            df = tables[0].df
            # skip col0 (symbols), grab *all* numeric cells on row 0
            vals = [
                df.iat[0, c].strip()
                for c in range(1, df.shape[1])
                if re.fullmatch(r"\d+\.\d+", df.iat[0, c].strip())
            ]
            if vals:
                logger.info("✔ %s [%s] → %s", pdf_path.name, flavor, vals)
                return vals
        except Exception as e:
            logger.warning("Camelot[%s] error on %s: %s", flavor, pdf_path.name, e)
    logger.warning("No numeric columns found by Camelot for %s", pdf_path.name)
    return []

def main():
    records = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    logger.info("Processing %d PDFs…", len(pdfs))
    for pdf in pdfs:
        vals = extract_from_pdf(pdf)
        if vals:
            rec = {"source_pdf": pdf.name}
            for i, v in enumerate(vals, start=1):
                rec[f"col{i}"] = v
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("✅ Saved %d rows to %s", len(df), OUTPUT_CSV)

if __name__ == "__main__":
    main()
