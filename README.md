# gpt_text_detector
Combining statistical and neural methods to improve machine text detection

## Installation
### With Conda
```shell
$ conda env create --file=env.yml
$ conda activate cs224n_project
```

## Running
tbd
### Test Baseline
```shell
python run.py --test --baseline
```
### Train Baseline
Before we proceed, first check if things work by doing by running
```shell
python run.py --train --baseline --epoch-size=5
```
Check to see if 
1. `logs` directory is generated with `tfevents` files and `best-model.pt`
2. Then try
    ```shell
   tensorboard --logdir=logdir
    ```
   This should ideally give you summary of train results. If not, skip it and move on. Clear the log files.

If all works fine, then run
```shell
python run.py --train --baseline
```
Then run ```tensorboad --logdir=logdir``` and save plots from UI. After that, you should remove the generated logfiles so that we can train the main model. Ideally, they should go in different directories, but I'm too lazy to code that up atm. 
### Test Trained Baseline
```shell
python run.py --test --baseline --from-checkpoint
```
This should output plots in the `results/` directory. Sanity check the plots. 
### Train Main Model
Likewise, check to see if everything works without fail
```shell
python run.py --train --epoch-size=5 --early-fusion --use-all-stats
```
Check the same things as above. If all works fine, then run
```shell
python run.py --train --early-fusion --use-all-stats
```
Then run ```tensorboad --logdir=logdir``` and save plots from UI. 

### Test Main Model
```shell
python run.py --test --early-fusion --use-all-stats
```
Check plots in `results/` directory and sanity check plots.
