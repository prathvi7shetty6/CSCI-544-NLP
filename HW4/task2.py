from conlleval import evaluate
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
import itertools
from collections import Counter
import numpy as np


vocab, embeddings = [], []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('glove.6B.100d.txt', 'rt', encoding='utf-8') as fi:
    full_content = fi.read().strip().split('\n')

for line in full_content:
    parts = line.split(' ')
    word = parts[0]
    embedding = [float(val) for val in parts[1:]]
    vocab.append(word)
    embeddings.append(embedding)

vocab = ['[PAD]', '[UNK]'] + vocab
pad_emb_npa = np.zeros((1, 100))  # embedding for '<pad>' token
unk_emb_npa = np.mean(embeddings, axis=0, keepdims=True)  # embedding for '<unk>' token

# Insert embeddings for pad and unk tokens at the top of embs_npa.
embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embeddings))

vocab_npa = np.array(vocab)
embs_npa = np.array(embs_npa)

print(len(embs_npa))
print(len(vocab_npa))

vocab_size = len(vocab_npa)

# NER tag mapping
ner_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8, '[PAD]': 9}
idx2tag = {idx: tag for tag, idx in ner_tags.items()}

dataset_glove = datasets.load_dataset("conll2003")

word_frequency = Counter()

word2idx_glove = {
    word: index
    for index, word in enumerate(vocab_npa)
}

# Iterating the dataset to replace unknown tokens with [UNK]
dataset_glove = (
    dataset_glove
    .map(lambda x: {
            'input_ids': [
                word2idx_glove.get(word.lower(), word2idx_glove['[UNK]'])
                for word in x['tokens']
            ]
        }
    )
)

# Removing columns pos_tags & chunk_tags; Renaming column ner_tags to labels
for split in dataset_glove.keys():
    dataset_glove[split] = dataset_glove[split].remove_columns(['pos_tags', 'chunk_tags'])
    dataset_glove[split] = dataset_glove[split].rename_column('ner_tags', 'labels')

train_data = dataset_glove['train']
validation_data = dataset_glove['validation']
test_data = dataset_glove['test']

class BiLSTMGlove(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, output_dim):
        super(BiLSTMGlove, self).__init__()
        droupout_val = 0.33

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), freeze=True)
        self.upper_embedding = nn.Embedding(2,10)
        self.lower_embedding = nn.Embedding(2,10)
        self.title_embedding = nn.Embedding(2,10)

        self.bilstm = nn.LSTM(embedding_dim+30, lstm_hidden_dim, num_layers = 1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(droupout_val)
        self.linear = nn.Linear(lstm_hidden_dim*2, output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, len(ner_tags)-1)

    def forward(self, x, is_upper, lower_case, title_case, labels=None):
        embed = self.embedding(x)
        upper = self.upper_embedding(is_upper)
        lower = self.lower_embedding(lower_case)
        title = self.lower_embedding(title_case)
        features = torch.cat((embed, upper, lower, title), dim=-1)
        lstm_out, _ = self.bilstm(features)
        drop = self.dropout(lstm_out)
        linear = self.linear(drop)
        elu_out = self.elu(linear)
        logits = self.classifier(elu_out)

        loss = None
        if labels is not None:
            logits_flatten = logits.view(-1, logits.shape[-1])
            labels_flatten = labels.view(-1)
            loss = nn.functional.cross_entropy(logits_flatten, labels_flatten,ignore_index=9)

        return logits, loss
        
def collate_fun_glove(batch):

    input_ids = [torch.tensor(item['input_ids'], device=device) for item in batch]
    labels = [torch.tensor(item['labels'], device=device) for item in batch]
    lengths = [len(label) for label in labels]

    # Additional features
    upper_case = [torch.tensor([1 if word.isupper() else 0 for word in item['tokens']], dtype=torch.long,device=device) for item in batch]
    lower_case = [torch.tensor([1 if word.islower() else 0 for word in item['tokens']], dtype=torch.long,device=device) for item in batch]
    title_case = [torch.tensor([1 if word.istitle() else 0 for word in item['tokens']], dtype=torch.long,device=device) for item in batch]

    upper_case_padded = torch.nn.utils.rnn.pad_sequence(upper_case, batch_first=True, padding_value=0)
    lower_case_padded = torch.nn.utils.rnn.pad_sequence(lower_case, batch_first=True, padding_value=0)
    title_case_padded = torch.nn.utils.rnn.pad_sequence(title_case, batch_first=True, padding_value=0)

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=9)

    return {'input_ids': input_ids_padded, 'labels': labels_padded, 'lengths': lengths, 'upper_case': upper_case_padded, 'lower_case': lower_case_padded,
            'title_case': title_case_padded}



batch_size = 32
train_loader = DataLoader(train_data, batch_size= batch_size, collate_fn = collate_fun_glove)
test_loader = DataLoader(test_data, batch_size= batch_size, collate_fn = collate_fun_glove)
validation_loader = DataLoader(validation_data, batch_size= batch_size, collate_fn = collate_fun_glove)

if __name__ == '__main__':
    model_glove = BiLSTMGlove(vocab_size=vocab_size, embedding_dim=100, lstm_hidden_dim=256, output_dim=128)
    # Model on test set
    model_glove.load_state_dict(torch.load('model_glove.pt',map_location=torch.device('cpu')))
    model_glove.eval()
    preds = []
    label_list = []
    with torch.no_grad():
        for data in test_loader:
            input_ids, labels, lengths, upper_case, lower_case, title_case = data['input_ids'], data['labels'], data['lengths'], data['upper_case'], data['lower_case'], data['title_case']
            logits, loss = model_glove(input_ids,  upper_case,  lower_case, title_case, labels)
            predictions = torch.argmax(logits, dim=2)

            for pred, label, length in zip(predictions.tolist(), labels.tolist(), lengths):
                decoded_label = [idx2tag[l] for l in label]
                label_list.extend([decoded_label[:length]])
                trimmed_pred = pred[:length]
                decoded_pred = [idx2tag[p] for p in trimmed_pred]
                preds.extend([decoded_pred])

    flat_preds = list(itertools.chain(*preds))
    flat_labels = list(itertools.chain(*label_list))
    precision, recall, f1 = evaluate(flat_labels, flat_preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
