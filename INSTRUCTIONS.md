# Instructions

## Dataset

The dataset is in ./data/ directory.
The script preprocess.sh prepares the data as follows:

- tokenize
- truecase
- BPE algorithm (split too long/rare words)

The files we are using: \*.bpe.tr \*.bpe.en and \*.tok.en.
Files:

- corpus - the training dataset
- newsdev2016 - devset
- newstest2016 - testset

Everything is preprocessed but all files are in here, just in case.

# Installation

You need:

1. CUDA 8.0
2. cuDNN6

This is what I've tested with Tensorflow, I had problems with newer CUDA9.0 and 9.1, so it's best to stick to this version. 

3. Tensorflow 1.4 (!)

OpenNMT works with this one, it's best not to use 1.5 yet.
Install it with:

pip install --upgrade tensorflow-gpu==1.4.0 --user

Check if it works with your CUDA etc by:
>> python
>> import tensorflow

If there are no errors, yay, it SHOULD be fine now.

4. OpenNMT

Clone it from:
https://github.com/OpenNMT/OpenNMT-tf

# Setting up the experiment

OpenNMT-tf uses 2 files to train the model:

- model.py - the architecture defined in the class
- config.yml - configuration settings

Copy ./config/ to your cloned ./OpenNMT/. It must be copied there because OpenNMT is silly and it doesn't like full paths, uh.

In each 'config.yml' you need to change filepaths to where you want the training to be saved.

I'm using simple filenames for experiments, so it's easier to navigate.

# Training

I'm not sure how to run the experiments in scheduled environment, sorry.

On a normal server/your computer:

See which GPUs are available with 'nvidia-smi' and set variable:

	cd ./OpenNMT-tf/

	CUDA_VISIBLE_DEVICES='0,1' python -m bin.main train --model config/models/baseline.py  --config config/baseline.yml

The training should start and you should see the log.

At the end of the training we may want to ensemble the last few checkpoints to average the weights, it gives better results:

	python -m bin.average_checkpoints --model_dir ../models/tensorflow/baseline --output_dir ../models/tensorflow/baseline/final_ensemble --max_count 3

## Monitoring training

I recommend using tensorboard, it helps to visualise training during/after it.


It installs with tensorflow and should work straight-away.
You can use multiple training directories at the same time, it's great to compare results.
Unfortunately, it doesn't allow to download the plots which sucks.
What I do, I download CVS files and plot them online with plot.ly because it's quick and efficient :) Then I download PNG and create vectorised PDF with it.

	python -m tensorboard.main --logdir=baseline:models/tensorflow/baseline,baseline+adam:models/tensorflow/baseline+adam,baseline+attention:models/tensorflow/baseline+attention,baseline+gru:models/tensorflow/baseline+gru

If you want to see it in the browser, connect to ssh with the server:

ssh -N -f -L localhost:16006:localhost:6006 user@server

Now you can see tensorboard via http://localhost:16006.

# Evaluation

The model has finished training (around 8 hrs on 1 GPU). Now what?

We want to finally evaluate the translation quality with BLEU score.

First, translate the testset.

	python -m bin.main infer --config config/baseline.yml --features_file ../data/newstest2016.bpe.tr --predictions_file ../models/tensorflow/baseline+adam/newstest2016.bpe.tr.translated --checkpoint_path ../models/tensorflow/baseline/final_ensemble/model.ckpt-80000


In ./models/, I've put scripts that are to be used for each model.
You should run the script from the directory, in which the model is located, with translated testset.

	sh -x validate.sh

You should see the score now. It is also saved to 'bleu' file in the same directory.
