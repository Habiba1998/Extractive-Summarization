from flask import Flask, request, jsonify
from bert import BertSummarizer
import os
model = BertSummarizer()
app = Flask(__name__)
#pipreqs --encoding utf-8 "./" --force
@app.route('/', methods=['POST'])
def predict():
    payload = request.get_json()
    body = payload["text"]
    num_sentences = payload["num_sentences"]
    summary = model.run(body = body, num_sentences=num_sentences)
    return jsonify({'summary': summary})

#if __name__ == "__main__":
#port = int(os.environ.get("PORT", 5000))

#for docker
app.run(host='0.0.0.0', port='5000')


#app.run(debug=True, host='0.0.0.0', port=port)
#app.run(debug=True, host='0.0.0.0', port='5000')
#file1 = open("hi.txt", "r", encoding="utf8")
#body = file1.read()
#summary = model.run(body = body, num_sentences=6)
#print(summary)
#file1.close()