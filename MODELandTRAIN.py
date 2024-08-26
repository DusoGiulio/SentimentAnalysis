import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import softmax
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  # Importa tqdm per la visualizzazione della barra di progresso

# Verifica se una GPU è disponibile e termina il programma se non è presente
if torch.cuda.is_available() == False:
    exit(2)

# Caricamento del modello e del tokenizer pre-addestrati
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)  # 4 classi: Neutral, Irrelevant, Positive, Negative

# Congela tutti gli strati del modello BERT eccetto gli ultimi due livelli
for name, param in model.bert.named_parameters():
    if not ('layer.11.' in name or 'layer.10.' in name):  # Solo gli ultimi due livelli
        param.requires_grad = False

# Verifica se una GPU è disponibile e imposta il dispositivo su 'cuda' se è disponibile, altrimenti su 'cpu'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)  # Sposta il modello sul dispositivo scelto

# Dataset personalizzato
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)  # Restituisce il numero totale di elementi nel dataset

    def __getitem__(self, idx):
        text = self.texts[idx]  # Estrae il testo dal dataset
        label = self.labels[idx]  # Estrae l'etichetta dal dataset
        encoded_input = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        input_ids = encoded_input['input_ids'].squeeze(0)  # Ottiene gli ID degli input
        attention_mask = encoded_input['attention_mask'].squeeze(0)  # Ottiene la maschera di attenzione
        label = torch.tensor(label, dtype=torch.long)  # Converte l'etichetta in un tensore
        return input_ids, attention_mask, label  # Restituisce gli ID degli input, la maschera di attenzione e l'etichetta

# Caricamento dei dataset CSV per il training e la validazione
train_df = pd.read_csv(r'data\twitter_training.csv')
val_df = pd.read_csv(r'data\twitter_validation.csv')

# Rimuove le righe con valori mancanti nella colonna 'tweet content'
train_df = train_df.dropna(subset=['tweet content'])
val_df = val_df.dropna(subset=['tweet content'])
 # Assicura che i contenuti dei tweet siano stringhe
train_df['tweet content'] = train_df['tweet content'].astype(str)
val_df['tweet content'] = val_df['tweet content'].astype(str)

# Mappatura delle etichette a valori numerici
sentiment_mapping = {'Neutral': 0, 'Irrelevant': 1, 'Positive': 2, 'Negative': 3}
# Converte le etichette in valori numerici per il training
train_df['sentiment'] = train_df['sentiment'].map(sentiment_mapping)
val_df['sentiment'] = val_df['sentiment'].map(sentiment_mapping)

# Creazione dei DataLoader per il training e la validazione
train_dataset = SentimentDataset(train_df['tweet content'].tolist(), train_df['sentiment'].tolist())
val_dataset = SentimentDataset(val_df['tweet content'].tolist(), val_df['sentiment'].tolist())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Impostazione dell'optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# Funzione di training
def train(model, train_loader, optimizer):
    model.train()  # Imposta il modello in modalità training
    total_loss = 0

    for input_ids, attention_mask, labels in tqdm(train_loader, desc="Training", leave=False):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # Azzeramento dei gradienti

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Esecuzione del forward pass
        loss = outputs.loss  # Ottiene la perdita
        total_loss += loss.item()  # Accumula la perdita

        loss.backward()  # Calcola i gradienti
        optimizer.step()  # Aggiorna i pesi

    avg_loss = total_loss / len(train_loader)  # Calcola la perdita media
    print(f'Loss media: {avg_loss}')

# Funzione di valutazione
def evaluate(model, val_loader):
    model.eval()  # Imposta il modello in modalità valutazione
    predictions, true_labels = [], []  # Liste per le previsioni e le etichette vere

    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for input_ids, attention_mask, labels in tqdm(val_loader, desc="Evaluation", leave=False):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)  # Esecuzione del forward pass
            logits = outputs.logits  # Ottiene le probabilità logit
            probs = softmax(logits, dim=1)  # Applica softmax per ottenere le probabilità
            pred_labels = torch.argmax(probs, dim=1)  # Ottiene le etichette previste

            predictions.extend(pred_labels.cpu().numpy())  # Aggiunge le previsioni alla lista
            true_labels.extend(labels.cpu().numpy())  # Aggiunge le etichette vere alla lista

    accuracy = accuracy_score(true_labels, predictions)  # Calcola l'accuratezza
    print(f'Accuratezza: {accuracy * 100:.2f}%')

    print(classification_report(true_labels, predictions, target_names=['Neutral', 'Irrelevant', 'Positive', 'Negative']))

# Funzione per controllare i gradienti dei parametri
def print_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parametro: {name}, Gradients: {param.grad is not None}")

# Training per diverse epoche
for epoch in range(10):  # Numero di epoche per l'addestramento
    print(f'Epoca {epoch + 1}')  # Stampa il numero dell'epoca corrente
    train(model, train_loader, optimizer)  # Esegue la funzione di training
    evaluate(model, val_loader)  # Esegue la funzione di valutazione
    print_gradients(model)  # Controlla i gradienti dopo ogni epoca

# Salva il modello fine-tuned e il tokenizer
model.save_pretrained('fine_tuned_bert_model')
tokenizer.save_pretrained('fine_tuned_bert_tokenizer')
