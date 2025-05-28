import os
import sys

current_path = os.path.abspath(".")
sys.path.append(f"{current_path}/src/explainers/lang_sam")
sys.path.append(f"{current_path}/src/explainers/lang_sam/segment_anything")
from .lang_sam import LangSAM
