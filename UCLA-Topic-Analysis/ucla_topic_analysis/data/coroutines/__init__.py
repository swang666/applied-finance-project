"""This module holds shared functions for coroutines.
"""

def print_progress(position, total):
    """This function prints out progress updates for the user.

    Args:
        position (int): How far along is the process
        total (int): The total number of steps.
    """
    percent = 100*position/total
    output = "progress: {percent:.2f}%".format(
        percent=percent)
    print(output, end="\r")

def create_file(file_path, initialdata=""):
    """Used to create a new file.

    Args:
        file_path (str): The path to the file for creation
        initialdata (str, optional): The data to add to the file on creation.
            Defaults to and empty string. If the file was not created then this
            is not used.

    Returns:
        boolean: True if the file was created and False if the file already
        existed
    """
    try:
        with open(file_path, "x") as new_file:
            new_file.write(initialdata)
    except FileExistsError:
        return False
    return True

def insert(text, file_to_update, position, block_size=64000):
    """This function is used for inserting data into a file.

    Args:
        text (str): The text to insert into the file
        file_to_update (:obj:`file`): The file to insert the text into
        position (int): The position in the file where the text needs to be
            inserted
        block_size (int, optional): The block size (in bytes) when reading the
            file. Defaults to 64000.
    """
    file_to_update.seek(position)
    while True:
        old_text = file_to_update.read(block_size)
        if old_text:
            file_to_update.seek(position)
            text += old_text
            file_to_update.write(text[:len(old_text)])
            text = text[len(old_text):]
            position = file_to_update.tell()
        else:
            file_to_update.write(text)
            break
