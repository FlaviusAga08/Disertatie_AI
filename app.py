from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)


model_dir = "./t5_legal_results"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)


model.eval()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]
        if not text.strip():
            return jsonify({"error": "Text is empty"}), 400

        inputs = tokenizer(
            "summarize: " + text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
