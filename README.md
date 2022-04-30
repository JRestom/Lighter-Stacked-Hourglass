# Lighter-Stacked-Hourglass
Modified stacked Hourglass architecture for Human pose estimation (HPE)


Human pose estimation (HPE) can be described as the process of approximating the configuration of the body from an image. To perform said task, models learn to detect and track semantic key-points (points of interest) on the human body, such as the knees, shoulders, and head, among others.
Here we focus on the approach proposed by Newell et al. [5], called the stacked-hourglass architecture, which uses several up-scaling and down-sampling blocks repeated
several times across the network to capture the local features and their positioning in the global context. Our goal is to modify the stacked-hourglass network (with two hour-glasses) and reduced the total number of parameters and improve the accuracy of the model.
