import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv('Suicide_Ideation_Dataset(Twitter-based).csv')  # Update with your dataset path
print(data.head())

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        text = text.lower()  # Lowercase
        # Add more preprocessing steps here if necessary
        # Todo: remove @, #, special charaters and words that not make any sense such as name
        return text
    return ''  # Return an empty string for non-strings

# Apply preprocessing
data['cleaned_text'] = data['Tweet'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
data['Suicide'] = label_encoder.fit_transform(data['Suicide'])  # The label column is named 'Suicide'

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_text'], data['Suicide'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = CountVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train).toarray()
X_test_vectorized = vectorizer.transform(X_test).toarray()
# Todo: visualize the frequency of words and vectors of word counds

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vectorized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_vectorized, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Assuming binary classification

    def forward(self, x):
        #Todo: activation function (optional)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleNN(input_size=X_train_tensor.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy:.4f}')

# Example prediction
def predict(text):
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text]).toarray()
    tensor_text = torch.tensor(vectorized_text, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(tensor_text)
        _, predicted = torch.max(output.data, 1)
    return label_encoder.inverse_transform(predicted.numpy())[0]

# Test the prediction function
example_text = "I'm feeling really hopeless and lost."
prediction = predict(example_text)
print(f'The predicted label for the text is: {prediction}')