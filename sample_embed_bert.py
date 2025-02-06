from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate embeddings
def get_embeddings(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state
    return embeddings

# Example usage
text = "Hello, world! This is a test for generating BERT embeddings."
embeddings = get_embeddings(text)

# Print the shape of the embeddings
print("Embeddings shape:", embeddings.shape)