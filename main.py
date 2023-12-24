from flask import Flask, jsonify, request
from collection_and_preprocessing.collection import GasDataCollector
from collection_and_preprocessing.preprocessing import GasPricePredictor

app = Flask(__name__)
api_key = "ffd3958a434b408b877819486ac689b8"
data_path = "./data/gas_dataset.csv"


@app.route('/predict', methods=['GET'])
def predict():
    collector = GasDataCollector(api_key)
    collector.collect_data()
    predictor = GasPricePredictor(data_path)
    df = predictor.load_and_preprocess_data()
    X_train, y_train, X_test, y_test = predictor.split_data(df)
    predictor.train_model(X_train, y_train)
    roc_auc, confusion_matrix, accuracy = predictor.evaluate_model(X_test, y_test)
    is_worth_send_now = bool(predictor.is_worth_send_now(df))
    return jsonify({
        "ROC AUC Score": roc_auc,
        "Confusion Matrix": confusion_matrix.tolist(),
        "Accuracy Score": accuracy,
        "Is worth send now": is_worth_send_now
    })


if __name__ == '__main__':
    app.run(debug=True)
