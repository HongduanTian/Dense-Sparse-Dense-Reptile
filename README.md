# Dense-Sparse-Dense Reptile
 This repository contains the code and experiments for the paper:  
 [Meta Learning with Network Pruning](http://arxiv.org/abs/2007.03219)
 [(ECCV '20)](https://eccv2020.eu/)
 
## Requirements
tensorflow-gpu=1.4  
python3.6  
tqdm  

## Getting the data  
### [miniimagenet dataset](https://drive.google.com/file/d/1GCozydCABFjbA7x7W7JJECy5GbCGForu/view?usp=sharing)

### tieredimagenet dataset  
Please download the compressed tar files from: https://github.com/renmengye/few-shot-ssl-public  
```
mkdir -p ../tieredImagenet/data  
tar -xvf tiered-imagenet.tar  
mv *.pkl ../tieredImagenet/data  
```

## Run experiments
For convenience, we provided 4 demos as seen in the repository. Directly run the .sh file or you can change some parameters in the file to run some other experiments.
```
sh ./[filename]
```
