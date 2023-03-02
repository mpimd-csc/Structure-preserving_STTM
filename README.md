# Efficient Structure-preserving Support Tensor Train Machine

This repository contains MATLAB files for the implementation of work proposed in the paper
 [Efficient Structure-preserving Support Tensor Train Machine](https://www.jmlr.org/papers/volume24/20-1310/20-1310.pdf).

**Intro** 

The key novelty of our research is a stable and well explained Support Vector Machine (SVM) model for low-rank tensor
input data that manifests much higher classification accuracy and banchmarked compared to other state-of-the-art methods.
Our paper presents a general SVM framework using the Tensor-Train decomposition 
along with the explanation, validation and importance of each stage of the proposed algorithm with a graphical illustration.



**Dataset**

```batch
Folder - datasets
```

We have taken two different types of datasets. One medical data (resting-state fMRI) and another Hyperspectral Images. 

*Medical resting-state fMRI Data*

 ADNI_first (Alzheimer disease) and ADHD (Attention Deficit Hyperactivity Disorder) 


*Hyperspectral Images*

Indian Pines and Salinas 


**Setup**

Libraries: 

1. [Tensor-Train Toolbox](https://github.com/oseledets/TT-Toolbox) by Ivan Oseledets and Sergey Dolgov 
2. [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) by Chih-Chung Chang and Chih-Jen Lin 


 
**Functions and Results**

Each folder presents results for each step of algorithm, presented in paper. 

<p float="left">
<img src="https://github.com/mpimd-csc/Structure-preserving_STTM/blob/main/Figure/CI95_TT-uTT_ADNI.png" width="200">
<img src="https://github.com/mpimd-csc/Structure-preserving_STTM/blob/main/Figure/CI95_TT-TTCP_ADNI.png" width="200">
<img src="https://github.com/mpimd-csc/Structure-preserving_STTM/blob/main/Figure/CI95_TTCP-TTCPe_ADNI.png" width="200">
<img src="https://github.com/mpimd-csc/Structure-preserving_STTM/blob/main/Figure/CI95_TTCPe-uTTCPe_ADNI.png" width="200">
</p>


Comparision of our method to state-of-the-art -> run the file named ``Mainfile_results.m`` in the 5th folder. 




**Cite As**

If you use our work and codes for the further research then please cite the paper [[Efficient_STTM]](https://arxiv.org/pdf/2002.05079.pdf).
<details><summary> BibTeX </summary><pre>
@article{JMLR:v24:20-1310,
  author  = {Kirandeep Kour and Sergey Dolgov and Martin Stoll and Peter Benner},
  title   = {Efficient Structure-preserving Support Tensor Train Machine},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {4},
  pages   = {1--22},
  url     = {http://jmlr.org/papers/v24/20-1310.html}
}
}</pre></details>

If you have any query/suggestion, kindly write to [Kirandeep Kour](https://www.mpi-magdeburg.mpg.de/person/59949/842836) at kour@mpi-magdeburg.mpg.de.
