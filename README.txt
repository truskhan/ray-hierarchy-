Welcome to the ray hierarchy project. The project enriches the PBRT renderer (Physicall Based Renderer from Mat Pharr and Greg Humphreys) with an accelerator - ray hierarchy and breadth-first renderer. The accelerator takes advantage of OpenCL parallel capability.

The OpenCL code could be run on CPU (thanks to ATI OpenCL SDK) or on GPU (use ATI or NVIDIA OpenCL SDK), in case you have compatible graphic card. 

The project aim is in exploring ray-hierarchy possibility in ray-object test speed up and compering the time spend in raytracing the scene with combine approach - using object space and ray space hierarchies. 

The idea and inspiration comes from several sources:
David Roger, Ulf Assarsson, and Nicolas Holzschuch. Whitted Ray-Tracing for Dynamic Scenes using a Ray-Space Hierarchy on the GPU,2007
http://artis.imag.fr/Publications/2007/RAH07/
A. J. Chung and A.J. Field. Ray space for hierarchical ray casting, 1999
ftp://ftp.computer.org/MAGS/CG&A/mms/111581.pdf
Laszlo Szecsi. The Hierarchical Ray Eengine, 2006
http://wscg.zcu.cz/wscg2006/papers_2006/full/g89-full.pdf
Kirill Garanzha and Charles Loop. Fast Ray Sorting and Breadth-First Packet Traversal for GPU Ray Tracing, 2010 
http://research.microsoft.com/en-us/um/people/cloop/garanzhaloop2010.pdf
