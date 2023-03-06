# ELIM_FER

> **Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition (NeurIPS 2022)**<br>

> [PAPER](https://arxiv.org/abs/2209.12172) | [DEMO](https://github.com/kdhht2334/ELIM_FER/tree/main/demo)

<a href="https://releases.ubuntu.com/16.04/"><img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-16.04-green"></a>
<a href="https://www.python.org/downloads/release/python-370/"><img alt="PyThon" src="https://img.shields.io/badge/Python-v3.8-blue"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>



[Daeha Kim](https://scholar.google.co.kr/citations?user=PVt7f0YAAAAJ&hl=ko), [Byung Cheol Song](https://scholar.google.co.kr/citations?user=yo-cOtMAAAAJ&hl=ko)

CVIP Lab, Inha University

<p align="center">
<img src="https://github.com/kdhht2334/ELIM_FER/blob/main/pics/main.png" height="200", width="3000"/>
</p>

### Update

- __2022.09.20__: Initialize this repository.


### Requirements

- Python (>=3.8)
- PyTorch (>=1.7.1)
- pretrainedmodels (>=0.7.4)
- [Wandb](https://wandb.ai/)
- [Fabulous](https://github.com/jart/fabulous) (terminal color toolkit)

To install all dependencies, do this.

```
pip install -r requirements.txt
```

### Datasets

1. Download four public benchmarks for training and evaluation (please download after agreement accepted).

  - [AffectNet](http://mohammadmahoor.com/affectnet/)
  - [Aff-wild](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) 
  - [Aff-wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
  - [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
 
 (For more details visit [website](https://ibug.doc.ic.ac.uk/))

2. Follow preprocessing rules for each dataset by referring pytorch official [custom dataset tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

### Training

Just run the below script!
```
chmod 755 run.sh
./run.sh <method> <gpu_no> <port_no> 
```
- `<method>`: `elim` or `elim_category`
- `<gpu_no>`: GPU number such as 0 (or 0, 1 etc.)
- `<port_no>`: port number to clarify workers (e.g., 12345)
* __Note__: If you want to try 7-class task (e.g., AffectNet), add `age_script` folder to your train or val. script and turn on `elim_category` option.

### Evaluation
- Evaluation is performed automatically at each `print_check` point in training phase.

### Demo

- Do to `demo` folder, and then feel free to use.
- Real-time demo with pre-trained weights
<p align="center">
<img src="https://github.com/kdhht2334/ELIM_FER/blob/main/demo/demo_vid.gif" height="320"/>
</p>


### TODO
- [x] Refactoring
- [ ] Upload pre-trained model weights
- [x] Upload demo files
- [x] Upload train/eval files


### Note
- In case of Mlp-Mixer, please refer official repository [(link)](https://github.com/google-research/vision_transformer#mlp-mixer)



### Citation

If our work is useful for your work, then please consider citing below bibtex:


	@misc{kim2022elim,
        author = {Kim, Daeha and Song, Byung Cheol},
        title = {Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition},
        Year = {2022},
        Eprint = {arXiv:2209.12172}
    }


### Contact
If you have any questions, feel free to contact me at `kdhht5022@gmail.com`.


