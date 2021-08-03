-----------------------------------------------------------------------------------------------------------------------
------------  cp3_alsls archive----------------------------------------------------------------------------------------
------------  CANDECOMP/PARAFAC Decomposition of a third order tensor via ALS coupled (or not) with Line Search ------
------------  @Copyright Dimitri Nion  -------------------------------------------------------------------------------
------------- Released May 2010 ---------------------------------------------------------------------------------------
------------  Feedback dimitri.nion@gmail.com -------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------

Computation of the CANDECOMP/PARAFAC Decomposition of a third-order tensor X via the Alternating Least Squares Algorithm (ALS), 
possibly combined with a Line Search technique to speed up convergence.
See header of the file for more details

Content of the pack: 
- Main function: cp3_alsls.m (all subfunctions nedeed are strapped in cp3_alsls.m)
- demo1, demo2, demo3, demo4: to show how to use the main function
- cp3_init.m : this is the same as the cp3_init.m subfunction strapped in cp3_alsls.m
It has been copied in the folder because it is called by demo1 and demo3
- solve_perm_scale.m : remove permutation and scaling ambiguity before computation of the error on the
loading matrices estimates
