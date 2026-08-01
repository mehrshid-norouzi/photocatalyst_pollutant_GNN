# from pathlib import Path
# import shutil
# # Project root
# PROJECT_ROOT = Path(__file__).resolve().parent

# # Data directories
# DATA_DIR = PROJECT_ROOT / "data"
# CIF_DIR = DATA_DIR / "cif_files"

# # Dataset files
# EXCEL_PATH = DATA_DIR / "dataset.xlsx"
# ATOM_INIT_PATH = DATA_DIR / "atom_init.json"

# # Output
# OUTPUT_DIR = PROJECT_ROOT / "output"

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def move_outputs_to_output_folder():
#     extensions = {
#         ".png",
#         ".jpg",
#         ".jpeg",
#         ".html",
#         ".xlsx",
#         ".xls",
#         ".csv",
        
#         ".json",
#         ".pth",
#         ".pt",
#         ".pkl",
#         ".pickle",
#     }

#     for file in PROJECT_ROOT.iterdir():

#         if file.is_file() and file.suffix.lower() in extensions:

#             destination = OUTPUT_DIR / file.name

#             try:
#                 if destination.exists():
#                     destination.unlink()

#                 shutil.move(str(file), str(destination))

#             except Exception as e:
#                 print(f"Could not move {file.name}: {e}")

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent

# ==================== Data ====================
DATA_DIR = PROJECT_ROOT / "data"
CIF_DIR = DATA_DIR / "cif_files"

EXCEL_PATH = DATA_DIR / "dataset.xlsx"
ATOM_INIT_PATH = DATA_DIR / "atom_init.json"


# ==================== Output ====================
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"

# Create folders
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==================== Move generated files ====================
def move_outputs_to_output_folder():

    figure_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".html",
    }

    result_extensions = {
        ".xlsx",
        ".xls",
        ".csv",
       
    }

    model_extensions = {
        ".pth",
        ".pt",
        ".pkl",
        ".pickle",
    }

    for file in PROJECT_ROOT.iterdir():

        if not file.is_file():
            continue

        suffix = file.suffix.lower()

        if suffix in figure_extensions:
            destination = FIGURE_DIR / file.name

        elif suffix in result_extensions:
            destination = RESULT_DIR / file.name

        elif suffix in model_extensions:
            destination = MODEL_DIR / file.name

        else:
            continue

        try:
            if destination.exists():
                destination.unlink()

            shutil.move(str(file), str(destination))

        except Exception as e:
            print(f"Could not move {file.name}: {e}")