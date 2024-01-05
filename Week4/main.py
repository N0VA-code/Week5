from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Flask 앱 생성
app = Flask(__name__)

# 모델 로드
model = pickle.load(open('logistic_regression_iris_model.pkl', 'rb'))



@app.route('/')
def index():
    return render_template('predict_form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 입력 데이터 추출
    data = request.form
    features = [float(data['sepal_length']), float(data['sepal_width']),
                float(data['petal_length']), float(data['petal_width'])]
    prediction = model.predict([features])

    # 예측 결과에 따른 종 이름 매핑
    species_dict = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    species = species_dict[prediction[0]]

    # 예측된 종의 이름 반환
    return jsonify({'species': species})


if __name__ == '__main__':
    app.run(debug=True)
