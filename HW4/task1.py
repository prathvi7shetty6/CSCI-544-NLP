from conlleval import evaluate
import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
import itertools
from collections import Counter

dataset = datasets.load_dataset("conll2003")

# Calculating the word frequency
word_frequency = Counter(itertools.chain(*dataset['train']['tokens']))
# Creating a dictionary with words having frequency greater than 2
word_frequency = {
    word: frequency
    for word, frequency in word_frequency.items()
    if frequency >= 3
}

# Adding the index and UNK and PAD to handle padding and unknown tokens:
word2idx = {
    word: index
    for index, word in enumerate(word_frequency.keys(), start=2)
}
word2idx['[PAD]'] = 0
word2idx['[UNK]'] = 1


# Iterating the dataset to replace unknown tokens with [UNK]
dataset = (
    dataset
    .map(lambda x: {
            'input_ids': [
                word2idx.get(word, word2idx['[UNK]'])
                for word in x['tokens']
            ]
        }
    )
)

# Removing columns pos_tags & chunk_tags; Renaming column ner_tags to labels
for split in dataset.keys():
    dataset[split] = dataset[split].remove_columns(['pos_tags', 'chunk_tags'])
    dataset[split] = dataset[split].rename_column('ner_tags', 'labels')
    
# NER tag mapping
ner_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8, '[PAD]': 9}
idx2tag = {idx: tag for tag, idx in ner_tags.items()}

vocab_size = len(word2idx)

train_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_hidden_dim, output_dim):
        dropout_val = 0.33
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers = 1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_val)
        self.linear = nn.Linear(lstm_hidden_dim*2, output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, len(ner_tags)-1)

    def forward(self, x, labels=None):
        embed = self.embedding(x)
        lstm_out, _ = self.bilstm(embed)
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
        
        
        
def collate_fun(batch):

    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    lengths = [len(label) for label in labels]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=9)
    return {'input_ids': input_ids_padded, 'labels': labels_padded, 'lengths': lengths}

batch_size = 32
train_loader = DataLoader(train_data, batch_size= batch_size, collate_fn = collate_fun)
test_loader = DataLoader(test_data, batch_size= batch_size, collate_fn = collate_fun)
validation_loader = DataLoader(validation_data, batch_size= batch_size, collate_fn = collate_fun)


if __name__ == '__main__':
    model = BiLSTM(vocab_size=vocab_size, embedding_dim=100, lstm_hidden_dim=256, output_dim=128)
    model.load_state_dict(torch.load('model.pt',map_location=torch.device('cpu')))
    model.eval()
    preds = []
    label_list = []
    with torch.no_grad():
        for data in test_loader:
            input_ids, labels, lengths = data['input_ids'], data['labels'], data['lengths']
            logits, loss = model(input_ids, labels)
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
    
