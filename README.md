# gpt_text_detector
123
Combining statistical and neural methods to improve machine text detection. 

## Installation
### With Conda
```shell
$ conda env create --file=env.yml
$ conda activate cs224n_project
```

### With Pip
```shell
$ pip install -r requirements.txt
```
## Experiments 
We recommend that you run the experiments in the following steps. All training results should be stored in `train_results/` and all test results should be stored in `test_results/`. The model checkpoints are saved in `mdl/`. The experiments default to the `gpt-wiki-intro` dataset. To use PubMedQA dataset, specify the following flag: `--datatype=pubmed_qa`.  
### Baseline Experiments
**1. Test Pre-trained Baseline**

This tests the "as-is" OpenAI RoBERTa detector (trained on GPT-2 data) on the given text.
```shell
$ python run.py --test --baseline
```
This runs a default test on the `gpt-wiki-intro` data set. 

**2. Finetune Baseline**

This finetunes the above baseline model on GPT-3 data. 
```shell
$ python run.py --train --baseline
```
This should save the model under `mdl/`. 

**3. Test Finetuned Baseline**

This tests the above finetuned baseline. NOTE: the model must be saved in the `mdl/` directory.
```shell
$ python run.py --test --baseline --from-checkpoint
```
### Main Model Experiments
**1. Train Main Model**

This tests the early fusion version of our main model. Note that you can turn on or off specific statistical features using flags, or use all as below. If `early_fusion` flag is not specified, model defaults to late fusion.
```shell
$ python run.py --train --early-fusion --use-all-stats
```

This should save the model under `mdl/`. 

**2. Test Main Model (Late Fusion)**

Test the model from above.
```shell
$ python run.py --test --early-fusion --use-all-stats
```
