# Lighter-Stacked-Hourglass
Modified stacked Hourglass architecture for Human pose estimation (HPE)


Human pose estimation (HPE) can be described as the process of approximating the configuration of the body from an image. To perform said task, models learn to detect and track semantic key-points (points of interest) on the human body, such as the knees, shoulders, and head, among others.
Here we focus on the approach proposed by Newell et al. [5], called the stacked-hourglass architecture, which uses several up-scaling and down-sampling blocks repeated
several times across the network to capture the local features and their positioning in the global context. Our goal is to modify the stacked-hourglass network (with two hour-glasses) and reduced the total number of parameters and improve the accuracy of the model.

## Architecture of a single hourglass
<img src="https://user-images.githubusercontent.com/99881055/166109645-586e3830-3ee8-4a40-8c4d-3d1a31c24755.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="500" height="250" />

## Results

<img src="https://user-images.githubusercontent.com/99881055/166109933-d365040f-1795-4564-9e13-42a2c450b961.png" data-canonical-src="https://gyazo.com/eb5c5741b6a9a16c692170a41a49c858.png" width="600" height="250" />

