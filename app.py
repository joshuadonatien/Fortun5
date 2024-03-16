from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/process_form', methods=['POST'])
def process_form():
    # Initialize the classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Collect form dat
    feelings = request.form['feelings']
    sleep = request.form['sleep']
    thoughts = request.form['thoughts']
    mood = request.form['mood']
    interactions = request.form['interactions']

    # Prepare data for processing
    user_inputs = {
        "feelings": feelings,
        "sleep": sleep,
        "thoughts": thoughts,
        "mood": mood,
        "interactions": interactions
    }
    
    # Process and classify each response
    results = {}
    for key, user_input in user_inputs.items():
        if user_input:  # Only process non-empty responses
            output = classifier(user_input, candidate_labels=["sad", "happy", "angry"])
            percentages = {label: f"{score * 100:.2f}%" for label, score in zip(output['labels'], output['scores'])}
            results[key] = percentages

    # Redirect to a new page or return results (for simplicity, redirecting back to home with results)
    # In practice, you might render a results page
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
