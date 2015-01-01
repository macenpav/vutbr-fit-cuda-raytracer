OVLADANI PROGRAMU
=================

Po prelozeni staci spustit bez parametru

Preklad
=======

Prilozeny projekt pro Microsoft Visual Studio 2013.
Potrebne zavislosti:
- C++11
- GLUT (https://www.opengl.org/resources/libraries/glut/), GLM (http://glm.g-truc.net/0.9.6/index.html), GLEW (http://glew.sourceforge.net/)
- CUDA 6.5 (https://developer.nvidia.com/cuda-downloads)

Nastaveni prekladu
==================
v constants.h:

dalsi vizualni moznosti:

OPT_BILINEAR_SAMPLING: akcelerace pomoci redukce vlaken a bilinearni samplovani meziprostoru
OPT_CAMERA_SHIFT: automaticky pohyb kamerou
OPT_DEPTH_OF_FIELD: hloubka ostrosti
OPT_SOFT_SHADOWS: mekke stiny

moznosti akcelerace:

ACC_NONE: bez akceleracni struktury
ACC_BVH: akcelerovano pomoci BVH (testovaci - obsahuje bugy)
ACC_KD_TREE: akcelerovano pomoci KD-tree