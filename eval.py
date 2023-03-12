import torch
from sklearn.metrics import roc_curve, auc, accuracy_score
from tqdm import tqdm


def compute_auroc(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def evaluate_model(model, test_loader, device):

    checkpoint = 1

    model.eval()

    test_loss = 0
    label_lst = []
    score_lst = []
    pred_lst = []
    with torch.no_grad(), tqdm(total=len(test_loader)) as progress_bar:
        for input_ids, masks, labels, stats, _ in test_loader:
            label_lst.extend(labels)
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), stats.to(
                device)  
            input_embeds = model.word_embeddings(input_ids)
            stat_embeds = None
            if not model.is_baseline:
                stat_embeds = model.stat_embeddings(stats)
                if model.is_early_fusion:
                    input_embeds += stat_embeds[:, None, :]
            loss, logits = model(inputs_embeds=input_embeds, attention_mask=masks, labels=labels,
                                 stat_embeds=stat_embeds, return_dict=False)

            test_loss += loss.item()
            p = logits.softmax(-1)
            pred_lst.extend(torch.argmax(p, dim=1).tolist())
            score_lst.extend(p[:, 1].tolist())  # probability of predicting label=1 (positive)

            if checkpoint % 100 == 0:
                print(f'\n\n===================CHECKPOINT {checkpoint}===================\n'
                      f'(test set) Accuracy: {accuracy_score(label_lst, pred_lst):.4f}\n'
                      f'(test set) AUROC: {compute_auroc(label_lst, score_lst)[2]:.4f}'
                      f'\n=========================END=========================\n')
            checkpoint += 1
            progress_bar.update(1)  # Update the tqdm progress bar

    accuracy = accuracy_score(label_lst, pred_lst)
    fpr, tpr, auroc = compute_auroc(label_lst, score_lst)
    test_loss /= len(test_loader.dataset)
    print(f'\n***********************FINALE****************************\n'
          f'(test set) Accuracy: {accuracy:.4f}\n'
          f'(test set) Average Loss: {test_loss:.4f}\n'
          f'(test set) AUROC Score: {auroc:.4f}\n'
          f'\n***************************************************\n')

    return test_loss, auroc, fpr, tpr, accuracy
