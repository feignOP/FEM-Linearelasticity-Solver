# add_paths.py
from pathlib import Path
import sys

def add_project_paths() -> None:
    """
    Add project subfolders (FemFrameWork, Kernel, DataFiles, Physics_models)
    to sys.path so they can be imported from anywhere.
    """
    root = Path(__file__).resolve().parent

    # folders we want to expose
    folders = [
        root / "FemFrameWork",
        root / "Kernel",
        root / "DataFiles",
        root / "Physics_models",   
    ]

    for folder in folders:
        folder_str = str(folder)
        if folder.exists() and folder_str not in sys.path:
            sys.path.insert(0, folder_str)

def load_data(filename: str):
    """Convenient loader for .npz files inside DataFiles."""
    data_path = Path(__file__).resolve().parent / "DataFiles" / filename
    import numpy as np
    return np.load(data_path)

# run on import
add_project_paths()

if __name__ == "__main__":
    print("Python path updated with project folders:")
    for p in sys.path[:5]:
        print("  ", p)