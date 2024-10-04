# intergrated-gradients-for-structured-pruning-in-bert
My thesis for MEng in NTUA


In this work, we have conducted experiments about Strucutred Pruning in BERT. Using interpretability techniques, especially for this work,
Integrated Gradients, we develop a novel metric based on spearman correlation, in order to extract an importance value for the self attention heads.

We evaluate our metric with Structured Pruning on classfication tasks of the GLUE benchmark for the BERT model.

We managed to accomplish good performance even in large prune ratios such as 80%.
