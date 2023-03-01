import torch
from sklearn.metrics import roc_auc_score
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
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), stats.to(
                device)  
            input_embeds = model.word_embeddings(input_ids)
            stat_embeds = None
            # TODO: Convert Stat Vector to input_embeds size
            if not model.is_baseline:
                stat_embeds = model.stat_embedding(stats)
                assert input_embeds.size() == stat_embeds.size()
                if model.early_fusion:
                    input_embeds += stat_embeds          
            output = model(input_embeds=input_embeds, attention_mask=masks, labels=labels, stat_embeds = stat_embeds)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(output[:, 1].cpu().numpy())  # Use the predicted probabilities for the positive class
            progress_bar.update(1)  # Update the tqdm progress bar

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    
    auroc = roc_auc_score(y_true, y_pred)
    print(f'Test set: AUROC score: {auroc:.4f}')

    return test_loss, accuracy, auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--early_fusion', action ="store_true")
    parser.add_argument('--baseline', action ="store_true")

    parser.add_argument('--all', '-a', action="store_true")
    parser.add_argument('--zipf', '-z', action="store_true")
    parser.add_argument('--heaps', '-e', action="store_true")
    parser.add_argument('--punctuation', '-p', action="store_true")
    parser.add_argument('--coreference', '-c', action="store_true")
    parser.add_argument('--creativity', '-r', action="store_true")

    args = parser.parse_args()

    d = {
        'zipf': args.all or args.zipf,
        'heaps': args.all or args.heaps,
        'punctuation': args.all or args.punctuation,
        'coreference': args.all or args.coreference,
        'creativity': args.all or args.creativity,
    }

    args.stat_extractor = StatFeatureExtractor(d)
    stat_size = args.stat_extractor.stat_vec_size()

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'

    _roberta = transformers.RobertaForSequenceClassification.from_pretrained(name)
    _model = RobertaWrapper(_roberta, stat_size, args.baseline, args.early_fusion)
    _loader = load_datasets(**vars(args))

    evaluate_model(_model, _loader, torch.nn.BCELoss, args.device)