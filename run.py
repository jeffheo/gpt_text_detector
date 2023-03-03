import os
import argparse
import torch
from torch.optim import Adam
import transformers
import matplotlib.pyplot as plt
from train import train, validate, RobertaWrapper
from eval import evaluate_model
from stat_extractor import StatFeatureExtractor
from dataset2 import load_datasets

logdir = "logs"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', '-t', action='store_true')
    parser.add_argument('--test', '-e', action='store_true')
    parser.add_argument('--baseline', '-b', action='store_true', help='run baseline ONLY')
    parser.add_argument('--from-checkpoint', action='store_true', help='load model from checkpoint')

    parser.add_argument('--large', '-l', action='store_true', help='use the roberta-large model instead of roberta-base')
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

    # model hyper-parameters
    parser.add_argument('--early-fusion', action="store_true")
    # TODO: late-fusion
    parser.add_argument('--unfreeze', action="store_true", help="unfreeze base RobertaForSequenceClassifier")
    parser.add_argument('--use-all-stats', action="store_true")

    # stat feature parameters
    # TODO: add more
    parser.add_argument('--zipf', '-z', action="store_true")
    parser.add_argument('--clumpiness', '-c', action="store_true")
    parser.add_argument('--punctuation', '-p', action="store_true")
    args = parser.parse_args()

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'
    args.tokenizer = transformers.RobertaTokenizer.from_pretrained(name)
    base = transformers.RobertaForSequenceClassification.from_pretrained(name)

    unfreeze = args.unfreeze

    stat_size = None
    if not args.baseline:
        args.stat_extractor = StatFeatureExtractor({
            'zipf': args.use_all_stats or args.zipf,
            'clumpiness': args.use_all_stats or args.clumpiness,
            'punctuation': args.use_all_stats or args.punctuation,
        })
        stat_size = args.stat_extractor.stat_vec_size
    else:
        if args.train:
            # if fine-tuning pre-trained roberta, unfreeze
            unfreeze = True

    model = RobertaWrapper(base, stat_size, unfreeze, args.baseline, args.early_fusion).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_loader, val_loader, test_loader = load_datasets(**vars(args))

    if args.train:
        print(f'Training {"BASE" if args.baseline else "MAIN"} Model...')
        os.makedirs(logdir, exist_ok=True)

        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(logdir)

        best_validation_accuracy = 0
        max_epochs = args.max_epochs or 1
        for epoch in range(1, max_epochs + 1):
            train_results = train(model, optimizer, train_loader, args.device, f'EPOCH {epoch}')
            validation_results = validate(model, val_loader, args.device)
            train_results["train/accuracy"] /= train_results["train/epoch_size"]
            train_results["train/loss"] /= train_results["train/epoch_size"]

            validation_results["val/accuracy"] /= validation_results["val/epoch_size"]
            validation_results["val/loss"] /= validation_results["val/epoch_size"]

            for key in train_results:
                key = key.split('/')[1]
                train_key = f'train/{key}'
                val_key = f'val/{key}'
                writer.add_scalar(train_key, train_results[train_key], global_step=epoch)
                writer.add_scalar(val_key, validation_results[val_key], global_step=epoch)

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
                    os.path.join(logdir, f"best-model-{'baseline' if args.baseline else 'main'}.pt")
                )

    elif args.test:
        # load model from checkpoint if it is main model, or if we're using fine-tuned baseline
        if not args.baseline or args.from_checkpoint:
            data = torch.load(os.path.join(logdir, f"best-model-{'baseline' if args.baseline else 'main'}.pt"))
            model.load_state_dict(data["model_state_dict"])

        loss, auroc, fpr, tpr, accuracy = evaluate_model(model, test_loader, args.device)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve (AUROC = {:.3f})'.format(auroc))

        model_name = ""
        if args.baseline:
            model_name = "baseline"
        elif args.early_fusion:
            model_name = "early_fusion"
        else:
            model_name = "late_fusion"
        plt.savefig(f'{model_name}_roc_curve.jpg')  # save the plot as a JPG file
        os.makedirs('results', exist_ok=True)
        results_path = os.path.join('results', f'{model_name}_evaluation_results.txt')
        with open(results_path, 'w+') as f:
            f.write(f'Loss: {loss:.4f}\n')
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write(f'AUC: {auroc:.4f}\n')
