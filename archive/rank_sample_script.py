#1. in run.py create LM object and assign it to args.rank_extractor, pass it into load_datasets
#2. for each text (in getitems) call check probabilities and retrieve the realtopk values for each token
#3. in getitem, for each token's rank, classify whether it's top 10 (0), top 100 (1), top 500 (2), top 1000 (3), top 5000 (4), and beyond (5)
#4. make getitem return this list
#5. have def rank_embeddings in the model class take in the rank list (available in the train/eval code that retrieves output from dataloader)
# and then process it from the rank embedding (just like def stat_embeddings)
#6  within the train loop, call rank_embeddings, get the rank embeddings and add it to the input embeddings 