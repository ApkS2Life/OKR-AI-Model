from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertModel, T5ForConditionalGeneration, T5Tokenizer, GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)

# Tải mô hình và tokenizer
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

@app.route('/generate_okr', methods=['POST'])
def generate_okr():
    data = request.json
    current_status = data['current_status']
    desired_goal = data['desired_goal']
    
    # Phân tích tình trạng hiện tại bằng DistilBERT
    inputs = bert_tokenizer(current_status, return_tensors='pt')
    outputs = bert_model(**inputs)
    bert_analysis = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    # Tạo OKR bằng T5
    t5_input_text = f"Tạo OKR từ: {current_status} đến {desired_goal}"
    t5_input_ids = t5_tokenizer.encode(t5_input_text, return_tensors='pt')
    t5_output_ids = t5_model.generate(t5_input_ids, max_length=200)
    t5_output = t5_tokenizer.decode(t5_output_ids[0], skip_special_tokens=True)

    # Tinh chỉnh chi tiết OKR bằng DistilGPT-2
    gpt2_input_text = f"Kế hoạch OKR: {t5_output}"
    gpt2_input_ids = gpt2_tokenizer.encode(gpt2_input_text, return_tensors='pt')
    gpt2_output_ids = gpt2_model.generate(gpt2_input_ids, max_length=300)
    gpt2_output = gpt2_tokenizer.decode(gpt2_output_ids[0], skip_special_tokens=True)

    return jsonify({
        'okr': gpt2_output
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
