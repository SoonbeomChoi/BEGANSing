# Korean Singing Voice Synthesis based on Auto-regressive Boundary Equilibrium GAN
Our research project develops a Korean singing voice synthesis system using deep neural network. The system generates singing voice from a given text and MIDI in an end-to-end manner. 

This repository contains PyTorch implementation for [Korean Singing Voice Synthesis based on Auto-regressive Boundary Equilibrium GAN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053950).

</p>

![model architecture final 2 3](https://user-images.githubusercontent.com/15067112/81911402-3917fe80-9608-11ea-9718-8a61b564a618.jpg)
<p align="center"><b>Overview of the proposed system</b></p>

# Pre-requisites
- Install PyTorch based on official website, https://pytorch.org/.
- Then install packages in requirements.txt.
```bash
pip install -r requirements.txt
```

**Tested environments**
<table>
    <tr>
        <th> Python </th>
        <th> CUDA </th>
        <th> PyTorch </th>
        <th> TorchAudio </th>
    </tr>
    <tr>
        <td> 3.6 ~ 3.8 </td>
        <td> 10.2 </td>
        <td> 1.5 </td>
        <td> 0.5 </td>
    </tr>
    <tr>
        <td> 3.7 </td>
        <td> 10.1 </td>
        <td> 1.5 </td>
        <td> 0.5 </td>
    </tr>
</table>

# Dataset
In the paper we used our own dataset and we plan to release the dataset. 
But if you want to use your own dataset, the dataset needs to meet few conditions as below or follow the the structure of 'sample_dataset'.

- Each song must have text(.txt), MIDI(.mid) and audio(.wav) and MIDI should be temporally aligned with audio.
- Text files assume to have the same number of syllables as MIDI notes. The text is aligned using MIDI note duration.
- Currently the system only supports Korean.

# Configuration
Check configuration files in 'config' folder.
- default_train.yml: Default configuration file for preprocess.py and train.py
- default_infer.yml: Default configuration file for infer.py

Change configurations before you run following steps and important parameters are as below.
- file_structure: File structure of dataset, 1: all the files in one folder, 2: .txt, .mid, .wav are in separated folders
- dataset_path: Path for dataset
- num_proc: The number of processes especially for preprocess.py
- use_cpu: Forcing code to use cpu and ignore 'device' parameter
- device: List of CUDA device indices (e.g. device: [0, 1] will use cuda:0 and cuda:1)
- batch_size: Training batch size
- data_mode: Dataloader mode, single: loading entire data on memory, multi: loading data with queue

# Preprocessing
```bash
python preprocess.py -c config/default_train.yml --use_cpu True
```

You can speed up preprocessing by increasing the number of processes or 'num_proc'.
You can use preprocess.py with GPU but 'num_proc' should be 1.

# Training
```bash
python train.py -c config/default_train.yml --device 0 --batch_size 32
```

If your system doesn't have enough memory, you can change 'data_mode' to 'multi' which loads data with queue.
Also type following command for tensorboard monitoring.
```bash
tensorboard --logdir=checkpoint/default
```

# Inference
```bash
python infer.py -c config/default_train.yml config/default_infer.yml --device 0
```

Specify text file and checkpoint file in the configuration and MIDI file is assumed to have same file name as text file. Together with train configuration, infer configuration is given then the configurations are merged.
You can also use pre-trained model from https://drive.google.com/file/d/1JqCD-kxba2meSlApWr06b2nzqLv3U-j_/view?usp=sharing.

# Results
Audio samples at: https://soonbeomchoi.github.io/saebyulgan-blog/. Model was trained at RTX2080 Ti with batch size 32 for 2 days.
![BEGANSing tensorboard](https://user-images.githubusercontent.com/15067112/82179828-34aa5900-991a-11ea-9f73-b85aad109cea.png)

# License 
- g2p/korean_g2p.py from https://github.com/scarletcho/KoG2P
- utils/midi_utils.py from Madmom, https://madmom.readthedocs.io/en/latest/

# To Do
- To publish pre-trained model on torchhub.
- To offer English based text interface for non-Korean speakers.
- To open the dataset used in the paper.