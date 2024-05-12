from flask import Flask,render_template,request
import pickle
app = Flask(__name__)

tokenizer  = pickle.load(open("models/cv.pkl","rb"))
model  = pickle.load(open("models/clf.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods = ["POST"])
def predict():
    email_text = request.form.get("email-content")
    ##tokenizing the email_text
    tokenized_email = tokenizer.transform([email_text])
    ##predictions
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions ==1 else -1
    return render_template("index.html", email_text=email_text, predictions = predictions)
if __name__ == "__main__":
    app.run(debug=True) 