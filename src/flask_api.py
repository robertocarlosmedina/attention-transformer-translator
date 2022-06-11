from flask_restful import Api, Resource, reqparse
from flask import Flask, jsonify
from flask_cors import CORS
import transformers
from src.tranformer import Transformer_Translator


app = Flask(__name__)
CORS(app, support_credentials=True)
api = Api(app)

request_put_args = reqparse.RequestParser()
request_put_args.add_argument("sentence", type=str, help="Sentece to be translated.")
transformer = Transformer_Translator()


class Translation(Resource):

    def post(self, source, target):
        # print(source, " => ",target)
        args = request_put_args.parse_args()
        valid = False
        source_sentence = args["sentence"]
        target_sentence = ""
        if source_sentence:
            valid = True
        target_sentence = transformer.get_translation(source_sentence)
        data = {"data": [{"translation": target_sentence,"valid": valid}]}
        return jsonify(data)


class Resfull_API:
    @staticmethod
    def start():
        api.add_resource(Translation, "/translate/<string:source>/<string:target>")
        app.run(debug=False)
