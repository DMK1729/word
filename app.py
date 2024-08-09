from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Ensure model runs on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def predict_next_word(input_text):
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(inputs.shape, device=device, dtype=torch.long)  # Create attention mask
    outputs = model.generate(
        inputs, 
        max_length=len(inputs[0]) + 1,  # Only generate one more token
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1  # Only need one sequence
    )
    next_token = outputs[0, len(inputs[0]):].tolist()  # Extract the next token
    predicted_text = tokenizer.decode(next_token, skip_special_tokens=True)
    return predicted_text.strip()  # Remove any surrounding whitespace

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')
    predicted_text = predict_next_word(input_text)
    return jsonify({'predicted_text': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)