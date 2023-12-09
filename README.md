# DeepIDA-GRU
Biomedical research now commonly integrates diverse data types or views from the same individuals to better understand the pathobiology of complex diseases, but the challenge lies in meaningfully integrating these diverse views. Existing methods often require the same type of data from all views (cross-sectional data only or longitudinal data only) or do not consider any class outcome in the integration method, presenting limitations. 

To overcome these limitations, we have developed a pipeline that harnesses the power of statistical and deep learning methods to integrate cross-sectional and longitudinal data from multiple sources. Additionally, it identifies key variables contributing to the association between views and the separation among classes, providing deeper biological insights. 

This pipeline includes **variable selection/ranking using linear and nonlinear methods**, **feature extraction using functional principal component analysis and Euler characteristics**, and **joint integration and classification using dense feed-forward networks and recurrent neural networks**.

Refer to the function **pipeline_example.py** for a demonstration of how to use this algorithm.

If you have any questions or suggestions for improvements, please email: ssafo@umn.edu 

Sarthak Jain and Sandra E. Safo **A  deep learning pipeline for  cross-sectional and longitudinal multiview data integration** (2023) Submitted

<img width="811" alt="DeepIDA_v4 (3)" src="https://github.com/lasandrall/DeepIDA-GRU/assets/29103607/d63183dc-3f08-4fb7-be42-5097761d0181">
