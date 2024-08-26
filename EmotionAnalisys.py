import tkinter as tk
from tkinter import ttk, messagebox
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd

# Caricamento del modello e del tokenizer pre-addestrati
model_name = "fine_tuned_bert_model"
tokenizer_name = "fine_tuned_bert_tokenizer"
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Caricamento dei dati di validazione
val_df = pd.read_csv(r'data\twitter_validation.csv')
val_df = val_df.dropna(subset=['tweet content'])
val_df['tweet content'] = val_df['tweet content'].astype(str)


sentiment_mapping = {'Neutral': 0, 'Irrelevant': 1, 'Positive': 2, 'Negative': 3}
reverse_sentiment_mapping = {v: k for k, v in sentiment_mapping.items()}
val_df['sentiment'] = val_df['sentiment'].map(sentiment_mapping)

# Funzione per ottenere la previsione del modello
def predict_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    with torch.no_grad():  # Disabilita il calcolo dei gradienti per la previsione
        outputs = model(input_ids, attention_mask=attention_mask)  # Esecuzione del forward pass
        logits = outputs.logits  # Ottiene le probabilità logit
        probs = torch.softmax(logits, dim=1)  # Applica softmax per ottenere le probabilità
        pred_label = torch.argmax(probs, dim=1).item()  # Ottiene l'etichetta prevista

    return reverse_sentiment_mapping[pred_label]  # Converte l'etichetta numerica in etichetta testuale

# Funzione per visualizzare il commento e la previsione
def show_prediction():
    try:
        idx = int(entry_index.get())  # Ottiene l'indice inserito dall'utente
        if idx < 0 or idx >= len(val_df):
            messagebox.showerror("Errore", "Indice fuori intervallo!")  # Mostra un messaggio di errore se l'indice è fuori intervallo
            return

        text = val_df.iloc[idx]['tweet content']  # Estrae il testo del tweet all'indice fornito
        true_label = reverse_sentiment_mapping[val_df.iloc[idx]['sentiment']]  # Ottiene l'etichetta vera
        predicted_label = predict_sentiment(text)  # Ottiene la previsione del modello

        result_text = f"Commento:\n{text}\n\n"  # Crea il testo del risultato
        result_text += f"Etichetta vera: {true_label}\n"
        result_text += f"Predizione del modello: {predicted_label}"

        text_result.delete('1.0', tk.END)
        text_result.insert(tk.END, result_text)

    except ValueError:
        messagebox.showerror("Errore", "Per favore, inserisci un indice valido.")

# Creazione della GUI
root = tk.Tk()
root.title("Sentiment Analysis")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


min_index = 0
max_index = len(val_df) - 1

label_index = ttk.Label(frame, text="Indice del commento:")
label_index.grid(row=0, column=0, sticky=tk.W)

entry_index = ttk.Entry(frame, width=10)
entry_index.grid(row=0, column=1, sticky=(tk.W, tk.E))

# Mostra il range degli indici
label_range = ttk.Label(frame, text=f"(Intervallo: {min_index} - {max_index})")
label_range.grid(row=0, column=2, sticky=tk.W)

button_predict = ttk.Button(frame, text="Mostra Predizione", command=show_prediction)
button_predict.grid(row=1, column=0, columnspan=3, pady=10)

text_result = tk.Text(frame, wrap='word', width=80, height=20)
text_result.grid(row=2, column=0, columnspan=3, pady=10)

root.mainloop()
