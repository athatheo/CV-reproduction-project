# F2DNet reproduction
This [paper](https://arxiv.org/pdf/2203.02331.pdf) was reproduced as part of the project in CS4245 course in TU Delft CS Masters.

You can find the original GitHub repository [here](https://github.com/AbdulHannanKhan/F2DNet).

We aim to reproduce the results independently, without use of the original code.

### Steps
1. Download the CityPersons dataset
   - The structure should be as shown below

```bash
├── datasets
│   ├── CityPersons
│     └── annotations
│         ├── anno_train.mat
│         └── anno_val.mat
│     └── images
│         ├── train
│         ├── val
│         └── test
```
2.




### Note 
DataLoading code taken from [lwpyr/CSP-pedestrian-in-pytorch](https://github.com/lwpyr/CSP-pedestrian-detection-in-pytorch/).
We plan to change that as well.

## HRNet backbone
### Dependencies:
pytorch(*the stable vision in official website is fine*)
The pretrained model should be downloaded from
https://drive.google.com/file/d/1NxCK7Zgn5PmeS7W1jYLt5J9E0RRZ2oyF/view
and put in the HRNet_pretrain/checkpoints folder

### Get started
Go to the HRNet_pretrain folder, run the *hrnetpretrain.py* file,
the input size is 2048*1024, and the output tensor size is [1, 480, 256, 128]