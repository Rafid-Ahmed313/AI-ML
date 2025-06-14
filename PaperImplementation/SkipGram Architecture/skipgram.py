import torch
import torch.nn as nn
import torch.optim as optim
import random

# Corpus
corpus = "I love natural language processing and I love deep learning".lower().split()

# Vocabulary
vocab = list(set(corpus))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

# Generate (center, context) pairs
def generate_training_data(corpus, window_size=2):
    pairs = []
    for i, center_word in enumerate(corpus):
        center_idx = word2idx[center_word]
        for j in range(-window_size, window_size + 1):
            if j == 0 or i + j < 0 or i + j >= len(corpus):
                continue
            context_word = corpus[i + j]
            context_idx = word2idx[context_word]
            pairs.append((center_idx, context_idx))
    return pairs

training_pairs = generate_training_data(corpus)

# SkipGram model
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.input_embed = nn.Embedding(vocab_size, embed_dim)
        self.output_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context):
        center_vec = self.input_embed(center)       # shape: [1, embed_dim]
        context_vec = self.output_embed(context)    # shape: [1, embed_dim]
        score = torch.sum(center_vec * context_vec, dim=1)  # Dot product
        log_prob = torch.log(torch.sigmoid(score))  # Log loss
        return -log_prob.mean()

# Training
embed_dim = 10
model = SkipGram(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    total_loss = 0
    random.shuffle(training_pairs)
    for center, context in training_pairs:
        center_tensor = torch.tensor([center], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)
        loss = model(center_tensor, context_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {total_loss:.4f}")

# Get learned embeddings
word_embeddings = model.input_embed.weight.data
print("Vector for 'love':", word_embeddings[word2idx['love']])

