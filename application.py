from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, predictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        else:
            # Collect form data
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score'),
            )

            pred_df = data.get_data_as_data_frame()
            print(pred_df)

            predict_pipeline = predictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Round and clamp manually
            results = round(results[0], 2)
            if results > 100:
                results = 100.00
            elif results < 0:
                results = 0.00

            return render_template('home.html', results=results)
    except Exception as e:
        print("Error occurred:", e)
        import traceback
        print(traceback.format_exc())
        return "Internal Server Error", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
  
