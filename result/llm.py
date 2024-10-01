import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go


# Load the dataset
df = pd.read_csv("D:\SKILL TRAINING\projects\LLM PROJECT\data\impression_300_llm.csv")

# Split the data into training and evaluation sets
train_text, eval_text, train_labels, eval_labels = train_test_split(df['Report Name'], df['History'], test_size=0.1, random_state=42)



# Load the pre-trained model and tokenizer
model_name = 'bert-base-uncased'  
model = AutoModelForSequenceClassification.from_pretrained(model_name)  
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Preprocess the text data
train_encodings = tokenizer.batch_encode_plus(train_text.tolist(),
                                              add_special_tokens=True,
                                              max_length=512,
                                              padding=True,
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='np')


eval_encodings = tokenizer.batch_encode_plus(eval_text.tolist(),
                                             add_special_tokens=True,
                                             max_length=512,
                                             padding=True,
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_tensors='np')



# Create a custom dataset class
class ImpressionDataset(torch.utils.data.Dataset):
 def __init__(self, encodings, labels):
  self.encodings = encodings
  self.labels = labels
 
 def __getitem__(self, idx):
  item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  item['labels'] = torch.nn.functional.one_hot(torch.tensor(self.labels[idx]),num_classes=len(set(self.labels)))
  return item
 def __len__(self):
  return len(self.labels)



# Create dataset instances
train_dataset = ImpressionDataset(train_encodings, train_labels)
eval_dataset = ImpressionDataset(eval_encodings, eval_labels)


# Create data loaders
batch_size = 16
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


# Fine-tune the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



for epoch in range(5):
 model.train()
 total_loss = 0
 for batch in train_dataloader:
  input_ids = batch['input_ids'].to(device)
  attention_mask = batch['attention_mask'].to(device)
  labels = batch['labels'].to(device)
  optimizer.zero_grad()
  outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
  loss = criterion(outputs, labels)
  loss.backward()
  optimizer.step()
  total_loss += loss.item()
  print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}')
model.eval()



# Generate impressions on the evaluation data
impressions = []
with torch.no_grad():
 for batch in eval_dataloader:
  input_ids = batch['input_ids'].to(device)
  attention_mask = batch['attention_mask'].to(device)
  outputs = model(input_ids, attention_mask=attention_mask)
  logits = outputs.logits
  impressions.extend(torch.argmax(logits, dim=1).cpu().numpy())



# Compute Perplexity and ROUGE score
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
perplexity = 0
rouge_scores = []
for i, impression in enumerate(impressions):
 perplexity += torch.exp(-torch.log(torch.tensor(impression)))
 rouge_scores.append(scorer.score(eval_text[i], impression))
perplexity /= len(impressions)
rouge_scores = pd.DataFrame(rouge_scores).mean()


print(f'Perplexity: {perplexity:.4f}')
print(f'ROUGE score: {rouge_scores:.4f}')


# Remove stop words and apply stemming and lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
 tokens = nltk.word_tokenize(text)
 tokens = [t for t in tokens if t not in stop_words]
 tokens = [lemmatizer.lemmatize(t) for t in tokens]
 return ' '.join(tokens)


df['Report Name'] = df['Report Name'].apply(preprocess_text)
df['History'] = df['History'].apply(preprocess_text)


# Convert text to embeddings
from transformers import AutoModel, AutoTokenizer
model_name = 'gemma-2b-it'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_embeddings(text):
 inputs = tokenizer.encode_plus(text,
                                add_special_tokens=True,
                                max_length=512,
                                return_attention_mask=True,
                                return_tensors='pt')
 outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
 embeddings = outputs.last_hidden_state[:, 0, :]
 return embeddings.detach().cpu().numpy()  


df['Report Name Embeddings'] = df['Report Name'].apply(get_embeddings)
df['History Embeddings'] = df['History'].apply(get_embeddings)


# Identify top 100 word pairs based on embedding similarity
from scipy.spatial.distance import cosine
distances = []
for i in range(len(df)):
 for j in range(i+1, len(df)):
  distance = cosine(df['Report Name Embeddings'][i], df['History Embeddings'][j])
  distances.append((distance, i, j))

distances.sort()
top_pairs = distances[:100]
print(top_pairs)


# Create a visualization of the top 100 word pairs
sns.set()
plt.figure(figsize=(10, 10))
for pair in top_pairs:
 plt.plot([pair[1], pair[2]], [pair[0], pair[0]], 'k-')
plt.xlabel('Word Index')
plt.ylabel('Cosine Similarity')
plt.title('Top 100 Word Pairs')
plt.show()


# Create an interactive visualization of the top 100 word pairs
fig = go.Figure(data=[go.Scatter(x=[pair[1] for pair in top_pairs],
                                 y=[pair[0] for pair in top_pairs],
                                 mode='lines',
                                 line=dict(color='black'))])  
fig.update_layout(title='Top 100 Word Pairs',
                  xaxis_title='Word Index',
                  yaxis_title='Cosine Similarity')  

fig.show()


