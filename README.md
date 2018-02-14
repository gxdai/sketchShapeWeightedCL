# Sketch-based shape retrieval with weighted contrastive loss


## Description

### Shape

* Shapes are rendered from 12 different angles. 
* All the rendered image are finetuned with AlexNet.
* Visual features are extracted in the second last layer.
* Max-pooling to get the final shape representation

### Sketch
* Simply fine-tuned with AlexNet and extract the visual features in the second last layer



## Run
102 denotes using 102 gpu, 0 denotes gpu index, train denotes the mode 
```
./runWasser.sh 102 0 train
```
102 denotes using 102 gpu, 0 denotes gpu index, evaluation denotes the mode 
```
./runWasser.sh 102 0 evaluation
```

