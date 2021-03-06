import argparse
from termcolor import colored

arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "blue_score",
        "meteor_score", "confusion_matrix", "count_parameters", "ter_score"
    ],
    help="Add an action to run this project"
)

arg_pr.add_argument(
    "-s", "--source", required=True,
    choices=[
        "en", "cv"
    ],
    help="Source languague for the translation"
)

arg_pr.add_argument(
    "-t", "--target", required=True,
    choices=[
        "en", "cv"
    ],
    help="Target languague for the translation"
)

args = vars(arg_pr.parse_args())


if args["source"] == args["target"]:
    print(
        colored("Error: Source languague and Target languague should not be the same.", "red", attrs=["bold"])
    )
    exit(1)


from src.transformer import Transformer_Translator


transformer_translator = Transformer_Translator(args["source"], args["target"])


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
        "blue_score": transformer_translator.calculate_blue_score,
        "meteor_score": transformer_translator.calculate_meteor_score, 
        "confusion_matrix": make_matrix_confusion,
        "count_parameters": transformer_translator.count_hyperparameters,
        "ter_score": transformer_translator.calculate_ter
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
