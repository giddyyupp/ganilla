import sys
from cx_Freeze import setup, Executable

includefiles = ['data/','models/','options/','scripts/','util/', 'test.py'] #folder,relative path. Use tuple like in the single file to set a absolute path.
packages  = ['torch', 'html.parser', 'tkinter', 'sys']

setup(
    name = "RootEnhance",
    version = "0.1",
    description = "BIG ROOT TOOL",
    options = {'build_exe': {'packages':packages,'include_files':includefiles}},
    executables = [Executable("GANILLAui.py", base = "Win32GUI")])
