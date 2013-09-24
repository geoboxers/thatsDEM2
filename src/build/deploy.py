import sys,os
import build
import shutil
import glob


ROOT_DIR=os.path.realpath(os.path.join(os.path.dirname(__file__),"../.."))
SRC_DIR=os.path.realpath(os.path.join(os.path.dirname(__file__),".."))

LIB_DIR=(os.path.join(ROOT_DIR,"lib"))
TRIANGLE_DIR=os.path.join(SRC_DIR,"triangle")
SLASH_DIR=os.path.join(SRC_DIR,"slash")
#UTILS_DIR=os.path.join(ROOT_DIR,"utils")
 

if not os.path.exists(LIB_DIR):
    os.mkdir(LIB_DIR)
else: 
    shutil.rmtree(LIB_DIR)


build.main(["",LIB_DIR])
#shutil.copy(os.path.join(SLASH_DIR,"slash.py"), os.path.join(BIN_DIR,"slash.py"))
#shutil.copy(os.path.join(TRIANGLE_DIR,"triangle.py"), os.path.join(BIN_DIR,"triangle.py"))



def loop_folder_and_copy(FOLDER_NAME):
    files=glob.glob(os.path.join(FOLDER_NAME,"*.py"))
    for name in files:
        shutil.copy(name,os.path.join(LIB_DIR,os.path.basename(name)))


loop_folder_and_copy(TRIANGLE_DIR)
loop_folder_and_copy(SLASH_DIR)
#loop_folder_and_copy(UTILS_DIR)

