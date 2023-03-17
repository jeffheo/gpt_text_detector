import os
import argparse
import torch
from torch.optim import Adam
import transformers
import matplotlib.pyplot as plt
from train import train, validate, RobertaWrapper
from eval import evaluate_model
from stat_extractor import StatFeatureExtractor
from dataset import load_datasets
from torchinfo import summary
import numpy as np
import datetime
import time
import json

checkpoint_dir = "mdl"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--inspect', '-i', action='store_true')

    parser.add_argument('--train', '-t', action='store_true')
    parser.add_argument('--test', '-e', action='store_true')
    parser.add_argument('--baseline', '-b', action='store_true', help='run baseline ONLY')
    parser.add_argument('--from-checkpoint', action='store_true', help='load model from checkpoint')

    parser.add_argument('--large', '-l', action='store_true',
                        help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=24)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None, help='size of dataset at each epoch')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='real')
    parser.add_argument('--fake-dataset', type=str, default='fake')
    parser.add_argument('--token-dropout', type=float, default=None)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--datatype', type=str, default='wiki_intro')

    # model hyper-parameters
    parser.add_argument('--early-fusion', action="store_true")
    # TODO: late-fusion
    parser.add_argument('--unfreeze', action="store_true", help="unfreeze base RobertaForSequenceClassifier")
    parser.add_argument('--use-all-stats', action="store_true")

    # stat feature parameters
    parser.add_argument('--zipf', '-z', action="store_true")
    parser.add_argument('--clumpiness', '-c', action="store_true")
    parser.add_argument('--punctuation', '-p', action="store_true")
    parser.add_argument('--burstiness', action="store_true")
    parser.add_argument('--kurt', action="store_true")
    parser.add_argument('--stopword-ratio', action='store_true')
    args = parser.parse_args()

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'
    args.tokenizer = transformers.RobertaTokenizer.from_pretrained(name)
    base = transformers.RobertaForSequenceClassification.from_pretrained(name)
    
    unfreeze = args.unfreeze
    
    print(args.early_fusion)

    stat_size = None
    if not args.baseline:
        args.stat_extractor = StatFeatureExtractor({
            'zipf': args.use_all_stats or args.zipf,
            'clumpiness': args.use_all_stats or args.clumpiness,
            'punctuation': args.use_all_stats or args.punctuation,
            'burstiness': args.use_all_stats or args.burstiness,
            'kurt': args.use_all_stats or args.kurt,
            'stopword_ratio': args.use_all_stats or args.stopword_ratio
        })
        stat_size = args.stat_extractor.stat_vec_size
    else:
        if args.train:
            # if fine-tuning pre-trained roberta, unfreeze
            unfreeze = True

    model = RobertaWrapper(base, stat_size, unfreeze, args.baseline, args.early_fusion).to(args.device)
    model_type = ''
    if args.baseline:
        if args.train or args.from_checkpoint:
            model_type = "Baseline_Finetuned"
        else:
            model_type = "Baseline_Pretrained"
    else:
        if args.early_fusion:
            model_type = "Main_Early_Fusion"
        else:
            model_type = "Main_Late_Fusion"

    if args.inspect:
        print(f'\n\n==================================================\n\n'
              f'            Inspecting {model_type} Model...'
              f'\n\n==================================================\n\n')
        summary(model)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader, val_loader, test_loader = load_datasets(**vars(args))

    # Once loaders are set, delete non-serializable items from args
    del args.__dict__['tokenizer']
    if args.__dict__.get('stat_extractor'):
        del args.__dict__['stat_extractor']

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    if args.train:
        print(f'\n\n==================================================\n\n'
              f'            Begin Training {model_type} Model...'
              f'\n\n==================================================\n\n')
        RESULTS_PATH = f'train_results/{model_type}/{START_DATE}-{START_TIME}'
        os.makedirs(RESULTS_PATH, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        with open(os.path.join(RESULTS_PATH, 'args.json'), 'w+') as f:
            json.dump(args.__dict__, f, indent=4)

        best_validation_accuracy = 0
        max_epochs = args.max_epochs or 1

        combined_results = {
            'train/accuracy': [0] * max_epochs,
            'train/loss': [0] * max_epochs,
            'val/accuracy': [0] * max_epochs,
            'val/loss': [0] * max_epochs,
            'time': [0] * max_epochs,
        }

        for epoch in range(1, max_epochs + 1):
            start = time.time()

            train_results = train(model, optimizer, train_loader, args.device, f'TRAINING EPOCH {epoch}...')
            validation_results = validate(model, val_loader, args.device)
            train_results["train/accuracy"] /= train_results["train/epoch_size"]
            train_results["train/loss"] /= train_results["train/epoch_size"]
            validation_results["val/accuracy"] /= validation_results["val/epoch_size"]
            validation_results["val/loss"] /= validation_results["val/epoch_size"]

            for k in train_results:
                k = k.split('/')[1]
                if k == "epoch_size":
                    continue
                combined_results[f'train/{k}'][epoch - 1] = train_results[f'train/{k}']
                combined_results[f'val/{k}'][epoch - 1] = validation_results[f'val/{k}']

            if validation_results["val/accuracy"] > best_validation_accuracy:
                best_validation_accuracy = validation_results["val/accuracy"]
                print("New Best Validation Accuracy: {:.4f}".format(best_validation_accuracy))
                model_to_save = model.module if hasattr(model, 'module') else model
                # Checkpoint best model
                torch.save(dict(
                    epoch=epoch,
                    model_state_dict=model_to_save.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    args=args
                ),
                    os.path.join(checkpoint_dir, f"{model_type}.pt")
                )

            end = time.time()
            print(f'\nFINISHED EPOCH {epoch} IN {end - start:.2f}s\n')
            combined_results['time'][epoch - 1] = end - start

        combined_results['total_runtime'] = sum([combined_results['time'][i] for i in range(max_epochs)])
        combined_results['avg_time_per_epoch'] = combined_results['total_runtime'] / max_epochs

        with open(os.path.join(RESULTS_PATH, 'train_results.json'), 'w+') as f:
            json.dump(combined_results, f, indent=4)

        epochs = np.arange(1, max_epochs + 1)
        plt.plot(epochs, combined_results['train/accuracy'], color='r')
        plt.xticks(np.arange(0, max_epochs + 1))
        plt.title("Training Accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        # plt.legend([p1, p2], ["train", "val"])
        plt.savefig(os.path.join(RESULTS_PATH, 'train_accuracy.jpg'))
        plt.close()

        plt.clf()

        plt.plot(epochs, combined_results['train/loss'], color='r')
        # (p2,) = plt.plot(epochs, combined_results['val/loss'], color='b')
        plt.xticks(np.arange(0, max_epochs + 1))
        plt.title("Training Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")

        # plt.legend([p1, p2], ["train", "val"])
        plt.savefig(os.path.join(RESULTS_PATH, 'train_loss.jpg'))
        plt.close()
        plt.clf()

    elif args.test:
        print(f'\n\n==================================================\n\n'
              f'            Begin Testing {model_type} Model...'
              f'\n\n==================================================\n\n')
        # load model from checkpoint if it is main model, or if we're using fine-tuned baseline
        if not args.baseline or args.from_checkpoint:
            data = torch.load(os.path.join(checkpoint_dir, f"{model_type}.pt"))
            model.load_state_dict(data["model_state_dict"])

        RESULTS_PATH = f'test_results/{model_type}/{START_DATE}-{START_TIME}'
        os.makedirs(RESULTS_PATH, exist_ok=True)

        with open(os.path.join(RESULTS_PATH, 'args.json'), 'w+') as f:
            json.dump(args.__dict__, f, indent=4)

        start = time.time()
        loss, auroc, fpr, tpr, accuracy = evaluate_model(model, test_loader, args.device)
        end = time.time()
        print(f'\nFINISHED TESTING IN {end - start:.2f}s\n')

        test_results = {
            "loss": loss,
            "auroc": auroc,
            "accuracy": accuracy,
            "total_runtime": end - start
        }

        with open(os.path.join(RESULTS_PATH, 'results.json'), 'w+') as f:
            json.dump(test_results, f, indent=4)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve (AUROC = {:.3f})'.format(auroc))

        plt.savefig(os.path.join(RESULTS_PATH, 'roc_curve.jpg'))  # save the plot as a JPG file
        plt.close()
        plt.clf()
