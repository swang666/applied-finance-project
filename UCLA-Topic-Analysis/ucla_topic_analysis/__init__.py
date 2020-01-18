"""Contains code needed by the whole project
"""
import os
import time
import configparser
from functools import wraps

def get_config():
    """This function retrieves the config stored in the config.ini file

    Returns:
        A configparser.ConfigParser object containing the configuration in the
        config.ini file
    """
    config = configparser.ConfigParser()
    file_dir = os.path.dirname(os.path.realpath(__file__))
    config.read(file_dir + "/../config.ini")
    return config

def get_workers():
    """This function returns the number for workers to use.

    Returns:
        int: The number of workers. Or Nonw if the value is not set or less than
        1.
    """
    workers = get_config().get("TRAINING", "workers")
    return int(workers) if workers is not None and int(workers) > 0 else None

def get_data_folder():
    """
    This function returns the path to the folder containing the financial
    texts

    Returns:
        str: The path to the data folder specified by the configuration
    """
    data_folder = get_config().get("DATA", "path")
    return os.path.expandvars(data_folder)

def get_filings_folder():
    """
    This function returns the path to the folder containing the financial
    filings(10k)

    Returns:
        str: The path to the data folder specified by the configuration
    """
    data_folder = get_config().get("FILINGS", "path")
    return os.path.expandvars(data_folder)

def get_file_list():
    """
    This function is used for getting the path to the pdfs in the financial
    folder

    Returns:
        list: A list of absolute paths for each file in DATA_FOLDER
        and its subfolders
    """
    data_folder = get_data_folder()

    file_list = []
    for dir_path, _dir_name, file_names in os.walk(data_folder):
        for file_name in file_names:
            if file_name[-4:] == ".txt":
                abs_path = os.path.join(dir_path, file_name)
                file_list.append(abs_path)
    return file_list

def log_async_time(func):
    """A decorator for logging the time it took for an async function to execute

    Args:
        func (function): The function to log the execution time for

    Returns:
        function: A wrapped function that will print out the time it took for
        the original function to run
    """
    name = func.__name__
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        delta_time = end_time - start_time
        print("-----")
        print("function `{0}` took {1:.2f} seconds to execute".format(name, delta_time))
        print("-----")
        return result
    return wrapper

def log_time(func):
    """A decorator for logging the time it took for a function to execute

    Args:
        func (function): The function to log the execution time for

    Returns:
        function: A wrapped function that will print out the time it took for
        the original function to run
    """
    name = func.__name__
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        delta_time = end_time - start_time
        print("-----")
        print("function `{0}` took {1:.2f} seconds to execute".format(name, delta_time))
        print("-----")
        return result
    return wrapper
