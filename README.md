# my-pytorch-learn-repo

This is a repository containing Jupyter notebooks for CPSC 8430 Homeworks

## HW1

### Installation

There is a requirements.txt file containing packages necessary to run the included Jupyter notebooks. These packages should be installed in a virtual environment using the following commands:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure the ipython kernel is part of the virtual environment.

```
pip install ipykernel
python -m ipykernel --name=hwenv
```

In Jupyter, set the kernel to hwenv.

Afterwards, you should be able to install and rerun any of the Jupyter notebooks.

### Files
- [Homework-part1](homework-part1.ipynb). This contains all section of HW 1-1
- [Homework-part2](homework-part2.ipynb). 
    - Visualize the optimization process.
    - Observe gradient norm during training
- [Homework-part2-gradientzero](homework-part2-gradzero.ipynb). What happens when gradient is almost zero.
- [Homework-part3-1](homework-part3-1.ipynb).  Can network fit random labels
- [Homework-part3-2](homework-part3-2.ipynb). Number of parameters vs generalization
- [Homework-part3-3-1](homework-part3-3-1.ipynb) Flatness vs generalization part 1
- [Homework-part3-3-2](homework-part3-3-2.ipynb) Flatness vs generalization part 2

## HW2

### Installation

Change directory to the hw2/hw2_1 directory
There is a requirements.txt file containing packages necessary to run HW2. These packages should be installed in a virtual environment using the following commands:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage
From the hw2/hw2_1 directory, run the hw2_seq2seq.sh file. If provides a usage string if incorrect number of arguments are provided.
```
./hw2_seq2seq.sh test_data output_testset.txt
python bleu_eval.py output_testset.txt > bleu_eval_results.txt
```

### Files
- [hw2_seq2seq.sh](hw2/hw2_1/hw2_seq2seq.sh) - Run the model against a test directory of videos to generate captions
- [model1.pth](hw2/hw2_1/model1.pth) - The model being executed
- [output_testset.txt](hw2/hw2_1/output_testset.txt) - A file generated by hw2_seq2seq.sh on the testing_data directory
- [train2.py](hw2/hw2_1/train2.py) - Usage: python train2.py. It generates 2 models with different hidden dimensions and runs for 100 epochs and saves the best fit model and the epoch where that best fit model was achieved to disk.
- [training_run.txt](hw2/hw2_1/training_run.txt) - Log file from execution of train2.py. It reports when the two models under training have their highest BLEU values in validation.
- [raw_data.txt](hw2/hw2_1/raw_data.txt) - A csv file of data from the training run.
- [output_testset.txt](hw2/hw2_1/output_testset.txt) - The output testset I generated to test the capability
- [bleu_eval_results.txt](hw2/hw2_1/bleu_eval_results.txt) - The bleu eval results I achieved from my output_testset.txt file.

## HW3

### Installation

Change directory to hw3. 
There is a requirements.txt file containing packages necessary to run HW3. These packages should be installed in a virtual environment using the following commands:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Usage
From the hw3 directory, the fine tuning can be rerun using the model2-1.py file. This is time consuming activity and averaged 50 minutes. The fine tuned model is listed in the files section below. The saved fine-tuned model can also be loaded directly.

### Files
- [model2-1.py](hw3/model2-1.py) - Model used to download the pre-trained model and fine tune it. It saves the fine tuned model to disk along with data from the epochs.
- [alert_finetuned2-1.pth](hw3/albert_finetuned2-1.pth) - The fine tuned model created in this exercise
- [raw_data2-1.txt](hw3/raw_data2-1.txt) - A csv file of data saved from fine tuned primarily loss and f1 scores 
- [output2-1.txt](hw3/output2-1.txt) - The output printed on command line when doing fine tuning
