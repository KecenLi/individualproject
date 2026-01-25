import sys
import os

# Get the absolute path to the project root (one level up from src)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
libs_path = os.path.join(project_root, 'libs')

if libs_path not in sys.path:
    sys.path.append(libs_path)

# Now standard imports like 'import advex_uar' will work if 'libs/advex_uar' exists
