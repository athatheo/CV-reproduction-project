## HRNet backbone
### Dependencies:
pytorch(*the stable vision in official website is fine*)
The pretrained model should be downloaded from
https://drive.google.com/file/d/1NxCK7Zgn5PmeS7W1jYLt5J9E0RRZ2oyF/view
and put in the HRNet_pretrain/checkpoints folder

### Get started
Go to the HRNet_pretrain folder, run the *hrnetpretrain.py* file,
the input size is 2048*1024, and the output tensor size is [1, 480, 256, 128]