from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

modelo_path = "/Users/liviagrigolon/Documents/GitHub/7-days-of-data-science/data/processed/modelo_svd.sav"

@app.route("/")
def home():
    return jsonify({"message": "API de recomendações de filmes com SVD"})

@app.route("/recomendar/<int:user_id>")
def get_recommendations(user_id):
    """Carregar o modelo dentro da função"""
    try:
        modelo_svd = joblib.load(modelo_path)
    except Exception as e:
        return jsonify({"erro": f"Erro ao carregar o modelo: {e}"})

    user_factors = modelo_svd["user_factors"]
    svd_components = modelo_svd["components"]
    ratings_matrix_index = modelo_svd["ratings_matrix_index"]
    ratings_matrix_columns = modelo_svd["ratings_matrix_columns"]

    user_index_map = {user_id: idx for idx, user_id in enumerate(ratings_matrix_index)}

    if user_id not in user_index_map:
        return jsonify({"erro": "Usuário não encontrado"})

    user_index = user_index_map[user_id]
    pred_ratings = np.dot(user_factors[user_index], svd_components)

    recomendacoes = pd.DataFrame({
        "movie_id": ratings_matrix_columns,
        "pred_rating": pred_ratings
    }).nlargest(10, "pred_rating")

    return jsonify(recomendacoes.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, port=8000)
