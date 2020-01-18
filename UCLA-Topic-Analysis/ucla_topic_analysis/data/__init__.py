"""This module contains shared functions that are needed during data processing.
"""
import os


MODULE_DIR = os.path.dirname(os.path.realpath(__file__))
TRAINING_FOLDER_PATH = os.path.join(MODULE_DIR, "training")

# Make the folder for intermediat training files if it does not exist
try:
    os.mkdir(TRAINING_FOLDER_PATH)
except FileExistsError as error:
    pass

def get_training_file_path(file_name):
    """This function is used to get the path to an intermediate training file.
    These are files that are created and used during the training of a model.

    Returns:
        str: the name of the model's file. It is of the form
        lda-num-topics.model
    """
    return os.path.normpath(os.path.join(TRAINING_FOLDER_PATH, file_name))
