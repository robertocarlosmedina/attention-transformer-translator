import argparse
import os

arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "flask_api", "blue_score",
        "meteor_score", "wer_score", "gleu_score", "matrix_confusion",
        "count_parameters"
    ],
    help="Add an action to run this project"
)
args = vars(arg_pr.parse_args())


from src.tranformer import Transformer
from src.flask_api import Resfull_API

def make_matrix_confusion() -> None:
    transformer.generate_confusion_matrix("Oi, manera ke bô ta?")


transformer = Transformer()
test_data = transformer.get_test_data()


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": transformer.console_model_test,
        "train": transformer.train_model,
        "test_model": transformer.test_model,
        "flask_api": Resfull_API.start,
        "blue_score": transformer.calculate_blue_score,
        "meteor_score": transformer.calculate_meteor_score, 
        "matrix_confusion": make_matrix_confusion,
        "count_parameters": transformer.count_hyperparameters
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
