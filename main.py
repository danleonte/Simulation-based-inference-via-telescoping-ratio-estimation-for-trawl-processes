# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:00:16 2024

@author: dleon
"""

import sys
from pathlib import Path


def setup_sys_path():
    """
    Import this and run the setup function from main.py in every script
    or notebook to make imports easy. 

    Example usage: 

    from main import setup_sys_path
    setup_sys_path()
    """
    PARENT_PATH = Path(__file__).parents[0]  # project/
    if PARENT_PATH not in sys.path:
        sys.path.append(str(PARENT_PATH))
