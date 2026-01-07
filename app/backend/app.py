from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from typing import Literal
from dotenv import load_dotenv
import joblib
import json
import numpy as np
import pandas as pd
from openai import OpenAI
import os
from textwrap import dedent

load_dotenv()

app = Flask(__name__)
CORS(app)

model = joblib.load('models/ridge_tuned_model.pkl')
scaler = joblib.load('models/scaler.pkl')

client = OpenAI(api_key=os.getenv("OPENAI_SECRET_KEY"))

with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

with open('models/encoding_info.json', 'r') as f:
    encoding_info = json.load(f)

class PredictionInput(BaseModel):
    study_hours: float = Field(..., ge=0, le=16, description="Study hours per day")
    class_attendance: float = Field(..., ge=0, le=100, description="Class attendance percentage")
    sleep_hours: float = Field(..., ge=0, le=12, description="Sleep hours per day")
    gender: Literal["male", "other", "female"]
    course: Literal["diploma", "bca", "b.sc", "b.tech", "bba", "ba", "b.com"]
    internet_access: Literal["yes", "no"]
    sleep_quality: Literal["poor", "average", "good"]
    study_method: Literal["coaching", "online videos", "mixed", "self-study", "group study"]
    facility_rating: Literal["low", "medium", "high"]
    exam_difficulty: Literal["hard", "moderate", "easy"]
    age: int = Field(..., ge=16, le=25, description="Age of the student")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = PredictionInput(**request.json)
        
        data_dict = input_data.model_dump()
        df = pd.DataFrame(0, index=[0], columns=feature_names)
        
        numeric_features = ['study_hours', 'class_attendance', 'sleep_hours']
        for feat in numeric_features:
            df[feat] = data_dict[feat]
        
        for cat_col in encoding_info['categorical_columns']:
            value = data_dict[cat_col]
            encoded_col = f"{cat_col}_{value.lower()}"
            if encoded_col in df.columns:
                df[encoded_col] = 1
        
        df_scaled = scaler.transform(df)
        prediction = float(np.clip(model.predict(df_scaled)[0], 0, 100))

        prompt = dedent(f"""
            You are an academic support assistant supporting teachers with evidence-informed guidance.

            You are given:
            1. A predicted exam score for a student.
            2. The student's responses to academic, behavioral, and contextual factors.
            3. A ranked list of factors that influence exam performance, ordered by importance.

            Your task is to generate concise, actionable recommendations that help teachers
            either strengthen consistency in strong areas or address gaps where improvement
            is most likely to increase exam performance.

            Important rules:
            - Base recommendations on the ranking of factor importance, not numeric model coefficients.
            - Avoid second-person language (do not address the student directly).
            - Focus on modifiable behaviors and learning conditions.
            - Avoid causal or guaranteed outcome claims.
            - Do NOT mention machine learning models, coefficients, SHAP, or regression.
            - Use supportive, constructive, and non-judgmental language.
            - Prioritize recommendations on factors that are BOTH high-impact AND currently suboptimal.
            - Explicitly acknowledge and reinforce strong performance where excellence is observed.
            - Do not generate recommendations based on age, gender, or course enrollment.
            - Contextual factors may be acknowledged to tailor advice, but must not drive prioritization.

            Guidance for identifying gaps and excellence:
            - Treat factors as "suboptimal" if they are low, inconsistent, or poorly rated relative to
            typical academic expectations.
            - Treat factors as "strong" if they are consistently high, well-structured, or well-supported.
            - If at least one high-impact factor is strong, include one recommendation focused on
            maintaining or reinforcing that behavior.
            - If all high-impact factors are strong, shift fully to reinforcement and consistency guidance
            rather than improvement.

            Ranked importance of factors (highest to lowest):
            1. Study hours
            2. Class attendance
            3. Study method (self-study, group study, online videos, mixed)
            4. Learning environment / facility quality
            5. Sleep duration
            6. Sleep quality
            7. Contextual factors (exam difficulty, internet access, course context, age) — use only to adapt recommendations

            Student input:
            - Predicted exam score: {prediction}
            - Study hours: {data_dict["study_hours"]}
            - Class attendance: {data_dict["class_attendance"]}
            - Primary study method: {data_dict["study_method"]}
            - Sleep hours: {data_dict["sleep_hours"]}
            - Sleep quality: {data_dict["sleep_quality"]}
            - Facility rating: {data_dict["facility_rating"]}
            - Internet access: {data_dict["internet_access"]}
            - Exam difficulty: {data_dict["exam_difficulty"]}
            - Course: {data_dict["course"]}
            - Age: {data_dict["age"]}

            Output requirements:
            - Provide 2–3 recommendations.
            - At least one recommendation must reinforce an existing strength if excellence is detected.
            - Each recommendation must target a specific area where improvement or consistency is relevant.
            - Order recommendations from highest to lowest expected impact.
            - Each recommendation should be one concise sentence.
            - Frame suggestions as opportunities for support and sustainability, not deficiencies.
            - Emphasize balanced and holistic development rather than isolated changes.

            Begin your response with one brief, encouraging summary sentence, followed by the recommendations as a bullet list.
            """
        )

        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        print(response.output_text)
        
        return jsonify({
            'predicted_score': prediction,
            'recommendations': response.output_text,
        })
    
    except PydanticValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation Error',
            'details': e.errors()
        }), 400
    
    except Exception as e:
        print(e)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
    }), 200
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
