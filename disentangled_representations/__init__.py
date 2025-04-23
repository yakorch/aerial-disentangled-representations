import pathlib

CODE_DIR = pathlib.Path(__file__).absolute().parent
PROJECT_DIR = CODE_DIR.parent

DATASETS_DIR = PROJECT_DIR / "datasets"
ORIGINAL_DATASETS_DIR = DATASETS_DIR / "original"
PREPROCESSED_DATASETS_DIR = DATASETS_DIR / "preprocessed"
