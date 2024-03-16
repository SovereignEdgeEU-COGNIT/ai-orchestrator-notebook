from ModelManager import ModelManager
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

modelManager: ModelManager

# Define a route to receive VM data
@app.route('/api/vm', methods=['POST'])
def receive_vm_data():
    data = request.json  # Parse JSON data from request
    # Validate or use the data as needed
    print("Received VM data:", data)

    # Example: you might want to pass this data to some ML function
    # result = some_ml_function(data)

    # For now, we just send back a confirmation response
    return jsonify({"status": "success", "message": "VM data received"}), 200

@app.route('/api/model', methods=['POST'])
def set_model():
    data = request.json
    model_type = data.get('model_type')
    model_name = data.get('model_name')
    try:
        modelManager.set_model(model_type, model_name)
        return jsonify({"status": "success", "message": "Model set"}), 200
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    modelManager = ModelManager()

    app.run(debug=True, port=port)
