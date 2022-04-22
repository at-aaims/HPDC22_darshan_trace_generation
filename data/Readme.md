This directory contains following files which are needed for training and the  trained model for trace generation.

* `appinfo_%s.npy`: numpy data with application names
* `fgen_%s_app%d.torch`: FeatureGenerator's trained model
* `tgen_%s.torch`: TraceGenerator's trained model

The list of data sets (scale-1, scale-2, and scale-3) and selected application names are as follows:

* Scale-1 (Huge)

     ID | NAME
    ----|----------------------------
     0  | PPP_Hierarchical.mpi.3d
     1  | cholla.paris-cuda
     2  | hacc_p3m
     3  | lbpm_random_force_simulator
     4  | main_parallel
     5  | nekrs
     6  | ramses3d
     7  | sigma.cplx.x
     8  | tusas
     9  | xgc-es-cpp-gpu

* Scale-2 (Large)

     ID | NAME
    ----|-----------------------------
     0  | hf_summit_nvblas.x
     1  | hf_summit_oblate.x
     2  | lalibe
     3  | main_parallel
     4  | pmemd.cuda.MPI
     5  | prog_ccm_ex_summit_nat.exe
     6  | python
     7  | s3d.x
     8  | xgc-es-cpp-gpu
     9  | xspecfem3D

* Scale-3 (Medium)

     ID | NAME
    ----|-----------------------------
     0  | dirac.x
     1  | epw.x
     2  | lalibe
     3  | ngp
     4  | rmg-gpu
     5  | s3d.x
     6  | xspecfem3D

