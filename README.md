# MitosisRegionOfInterest

This is code + data to our manuscript submission entitled "Determination of the Mitotically Most Active Region for Computer-Aided Mitotic Count".

Summary:
   As part of many tumor grading schemes, the task considered was to find the mitotically most active region of a tumor. The properties of this region, especially the mitotic count, are related to the malignancy of the tumor. Thus, finding this region of interest is the first step in manual tumor grading by an expert pathologist. However, since time is limited in a routine diagnostic setting, this is a challenging task for human experts, as shown by our paper. We thus compared three deep learning-based approaches all using state-of-the-art architectures to fulfill this task. 
   

Overview:

- In the Methods folder, all implementations (regression, object detection, segmentation) are available.
- The folder results/ contains all our results for your reference
- The folder data/ contains the database with the anonymized expert opinion about the field of interest. 
- The notebook Results.ipynb contains the code for the generation of the result plots of the manuscript, as well as for the statistical analysis.


Note that this work is using the MITOS_WSI_CCMCT data set, available as open data. Please refer to the following repository to obtain it:

https://github.com/maubreville/MITOS_WSI_CCMCT

The scripts in this repo expect the data set to be present, and the Setup.ipynb notebook to be run to import all the data from figshare.
