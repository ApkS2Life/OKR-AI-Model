from flask import Flask, request, jsonify
import spacy
from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification,
                           RobertaTokenizer, RobertaForSequenceClassification,
                           DebertaTokenizer, DebertaForSequenceClassification,
                           T5Tokenizer, T5ForConditionalGeneration)
import torch

# Khởi tạo Flask app
app = Flask(__name__)

# Tải mô hình spaCy
nlp_spacy = spacy.load("en_core_web_sm")

# Tải mô hình BERT
tokenizer_bert = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model_bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Tải mô hình RoBERTa
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
model_roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Tải mô hình DeBERTa
tokenizer_deberta = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
model_deberta = DebertaForSequenceClassification.from_pretrained("microsoft/deberta-base")

# Tải mô hình T5
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text):
    inputs = tokenizer_t5(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model_t5.generate(inputs["input_ids"], max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer_t5.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).item()
    return predictions

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    # spaCy
    doc = nlp_spacy(text)
    spacy_analysis = {
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'pos_tags': [(token.text, token.pos_) for token in doc]
    }

    # BERT
    bert_prediction = process_text(text, model_bert, tokenizer_bert)

    # RoBERTa
    roberta_prediction = process_text(text, model_roberta, tokenizer_roberta)

    # DeBERTa
    deberta_prediction = process_text(text, model_deberta, tokenizer_deberta)

    # T5 Summary
    t5_summary = summarize_text(text)

    return jsonify({
        'spacy_analysis': spacy_analysis,
        'bert_prediction': bert_prediction,
        'roberta_prediction': roberta_prediction,
        'deberta_prediction': deberta_prediction,
        't5_summary': t5_summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
