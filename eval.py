import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse 
from dataset import load_datasets
from train import RobertaWrapper
import transformers

from stat_extractor import StatFeatureExtractor


def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad(), tqdm(total=len(test_loader)) as progress_bar:  # Disable gradient calculation during inference and add tqdm progress bar
        for input_ids, masks, labels, stats in test_loader:
            # stats = torch.tensor(stats).float()
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), stats.to(
                device)  
            input_embeds = model.word_embeddings(input_ids)
            stat_embeds = None
            if not model.is_baseline:
                stat_embeds = model.stat_embeddings(stats)
                # assert input_embeds.size() == stat_embeds.size()
                if model.early_fusion:
                    input_embeds += stat_embeds
            loss, logits = model(inputs_embeds=input_embeds, attention_mask=masks, labels=labels, stat_embeds=stat_embeds, return_dict=False)
            test_loss += loss
            pred = logits.argmax(dim=1, keepdim=True)
            logits = logits.detach().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(logits)  # Use the predicted probabilities for the positive class
            progress_bar.update(1)  # Update the tqdm progress bar

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    auroc = roc_auc_score(y_true, y_pred)
    print(f'Test set: AUROC score: {auroc:.4f}')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return test_loss, accuracy, auroc, fpr, tpr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='real')
    parser.add_argument('--fake-dataset', type=str, default='fake')
    parser.add_argument('--token-dropout', type=float, default=None)
    parser.add_argument('--early_fusion', action ="store_true")
    parser.add_argument('--baseline', action ="store_true")

    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--no_freeze', action='store_true')

    parser.add_argument('--all', '-a', action="store_true")
    parser.add_argument('--zipf', '-z', action="store_true")
    parser.add_argument('--clumpiness', '-l', action="store_true")
    parser.add_argument('--punctuation', '-p', action="store_true")
    parser.add_argument('--coreference', '-c', action="store_true")
    parser.add_argument('--creativity', '-r', action="store_true")

    parser.add_argument('--baseline_only', action='store_true')  # run baseline ONLY, default False

    args = parser.parse_args()

    d = {
        'zipf': args.all or args.zipf,
        'clumpiness': args.all or args.clumpiness,
        'punctuation': args.all or args.punctuation,
        'coreference': args.all or args.coreference,
        'creativity': args.all or args.creativity,
    }

    args.stat_extractor = StatFeatureExtractor(d)
    stat_size = args.stat_extractor.stat_vec_size

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'
    tokenizer = transformers.RobertaTokenizer.from_pretrained(name)
    args.tokenizer = tokenizer

    model_name = ""
    if args.baseline:
        model_name = "baseline"
    elif args.early_fusion:
        model_name = "early_fusion"
    else:
        model_name = "late_fusion"
    
    _roberta = transformers.RobertaForSequenceClassification.from_pretrained(name)
    _model = RobertaWrapper(_roberta, stat_size, args.baseline, args.early_fusion)

    params_path = os.path.join('gpt_text_detector/params', f'{model_name}_parameters.pth')
    _model.load_state_dict(torch.load(params_path))  
    
    _, _loader = load_datasets(**vars(args))
    
    loss, accuracy, auroc, fpr, tpr = evaluate_model(_model, _loader, torch.nn.BCELoss, args.device)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--') # diagonal line
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (AUROC = {:.3f})'.format(auroc))

    plt.savefig(f'{model_name}_roc_curve.jpg') # save the plot as a JPG file

    plt.show()
    results_path = os.path.join('gpt_text_detector/results', f'{model_name}_evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f'Loss: {loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'AUC: {auroc:.4f}\n')
