import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")  # ton fichier Firebase
firebase_admin.initialize_app(cred)
db = firestore.client()
from flask import Flask, request, jsonify
from model_loader import encode
from sentence_transformers import util

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_texts():
    data = request.get_json()
    desc1 = data.get('description1')
    desc2 = data.get('description2')

    if not desc1 or not desc2:
        return jsonify({'error': 'Missing descriptions'}), 400

    emb1 = encode(desc1)
    emb2 = encode(desc2)

    similarity = util.pytorch_cos_sim(emb1, emb2).item()

    return jsonify({'similarity_score': round(similarity, 2)})

@app.route('/report-Lost', methods=['POST'])
def report_found():
    data = request.get_json()
    description = data.get("description")

    if not description:
        return jsonify({"error": "Description is required"}), 400

    # Générer l’embedding
    embedding = encode(description)
    embedding_list = embedding.tolist()

    # Ajouter l’embedding aux données
    data["embedding"] = embedding_list

    # Enregistrer dans Firestore
    db.collection("lostObjects").add(data)

    return jsonify({"message": "Objet perdu enregistré avec embedding "}), 200
@app.route('/report-found', methods=['POST'])
def report_found():
    data = request.get_json()
    description = data.get("description")

    if not description:
        return jsonify({"error": "Description is required"}), 400

    # Générer l’embedding
    embedding = encode(description)
    embedding_list = embedding.tolist()

    # Ajouter l’embedding aux données
    data["embedding"] = embedding_list

    # Enregistrer dans Firestore
    db.collection("foundObjects").add(data)

    return jsonify({"message": "Objet trouvé enregistré avec embedding "}), 200
if __name__ == '__main__':
    app.run(debug=True)