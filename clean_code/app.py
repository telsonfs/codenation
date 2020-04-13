from flask import Flask, request, jsonify
from loguru import logger
from statistics import mean

from statsapi import data_store

app = Flask(__name__)


@app.route('/data', methods=['POST'])
def save_data():
    logger.info(f'Saving data...')

    content = request.get_json()
    uuid = data_store.save(content['data'])
    logger.info(f"Data saved with UUID '{uuid}' sucessfuly")

    return jsonify({"status": "sucess", "message": "data saved sucessfuly", "uuid": uuid})


@app.route('/data/<uuid>', methods=["GET"])
def retrieve_data(uuid):
    logger.info(f"Retrieving data associated with UUID '{uuid}'")
    try:
        data = data_store.get(uuid)

    except KeyError:
        logger.warning(f"cannot retrieve data associated with UUID '{uuid}'")
        return jsonify({"status": "failed", "message": "data cannot be retrieved", "data": []})

    logger.info(f"Data associated with UUID '{uuid}' retrieved sucessfully")

    return jsonify({"status": "success", "message": "data retrieved sucessfully", "data": data})


@app.route('/data/<uuid>/<operation>', methods=['GET'])
def operation_data(uuid, operation):
    list_operations = ['max', 'min', 'mean']
    if operation not in list_operations:
        return jsonify({"status": "failed", "message": "invalid operation", "data": []})

    logger.info(f"Retrieving data associated with UUID '{uuid}'")

    try:
        data = data_store.get(uuid)

    except KeyError:
        logger.warning(f"cannot retrieve data associated with UUID '{uuid}'")
        return jsonify({"status": "failed", "message": "data cannot be retrieved", "data": []})

    if operation == 'max':
        return jsonify({"status": "success", "message": "operation successfuly", "max": max(data)})
    elif operation == 'min':
        return jsonify({"status": "success", "message": "operation successfuly", "max": min(data)})
    elif operation == 'mean':
        return jsonify({"status": "success", "message": "operation successfuly", "mean": mean(data)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
