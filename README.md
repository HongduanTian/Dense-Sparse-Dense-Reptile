# Dense-Sparse-Dense Reptile
 This repository contains the code and experiments for the paper:  
 [Meta Learning with Network Pruning](https://openreview.net/pdf?id=FD8fE16Zo)
 [(ECCV '20)](https://eccv2020.eu/)
 
## Requirements
tensorflow-gpu=1.4  
python3.6  
tqdm  

## Getting the data  
### miniimagenet dataset
Run [fetch_data.sh](fetch_data.sh) script to acquire the dataset needed in the experiments. The script file creates a `data/` directory and downloads miniimagenet into it.
The size of dataset is about 5GB, so it takes about 10-20 min to download this dataset on a reasonably fast internet connection.

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
