import argparse
# import os

arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "flask_api", "blue_score",
        "meteor_score", "matrix_confusion", "count_parameters", "ter_score"
    ],
    help="Add an action to run this project"
)
args = vars(arg_pr.parse_args())


from src.tranformer import Transformer_Translator
from src.flask_api import Resfull_API


transformer_translator = Transformer_Translator()
test_data = transformer_translator.get_test_data()


def make_matrix_confusion() -> None:
    sentence = input("  Your Sentence: ")
    transformer_translator.generate_confusion_matrix(sentence)


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": transformer_translator.console_model_test,
        "train": transformer_translator.train_model,
        "test_model": transformer_translator.test_model,
        "flask_api": Resfull_API.start,
        "blue_score": transformer_translator.calculate_blue_score,
        "meteor_score": transformer_translator.calculate_meteor_score, 
        "matrix_confusion": make_matrix_confusion,
        "count_parameters": transformer_translator.count_hyperparameters,
        "ter_score": transformer_translator.calculate_ter
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
