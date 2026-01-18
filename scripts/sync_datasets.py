
import logging
from pathlib import Path
from kladml.backends import get_metadata_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync_datasets")

def sync():
    """Scan data/datasets and register them in the DB."""
    data_path = Path("data/datasets")
    if not data_path.exists():
        logger.info("data/datasets directory not found.")
        return

    metadata = get_metadata_backend()
    
    # Iterate top-level subdirectories (dataset names)
    for d in data_path.iterdir():
        if d.is_dir():
            name = d.name
            path = str(d)
            logger.info(f"Found dataset on disk: {name}")
            
            # Create in DB
            metadata.create_dataset(
                name=name,
                path=path,
                description=f"Imported from {path}"
            )
            logger.info(f"  -> Synced to DB.")

if __name__ == "__main__":
    sync()
