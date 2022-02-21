import logging
import os
import shutil
import sys
from unicodedata import name

def go_up(level_up):
    '''
    Simple function that returns the path of parent directory at a specified level

    Parameters
    ----------
    level up : int
        How much level you want to go up

    Returns
    --------
    path : str
        Path of the directory at level you chose from the current directory.
    '''
    if level_up == 0:
        path = os.getcwd()
    if level_up == 1:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    if level_up == 2:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    if level_up == 3:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    if level_up == 4:
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
    return path


def smart_makedir(name_dir, level_up=0):
    '''
    Easy way to create a folder. You can go up from the current path up to 4 times.

    Parameters
    ----------
    name_dir : str
        From level you have set, complete path of the directory you want to create
    level_up : int, optional
        How many step up you want to do from the current path. The default is 0.

    Returns
    -------
    None.
    '''
    separator = '/'
    if level_up == 0:
        path = separator.join([go_up(0), name_dir])
    if level_up == 1:
        path = separator.join([go_up(1), name_dir])
    if level_up == 2:
        path = separator.join([go_up(2), name_dir])
    if level_up == 3:
        path = separator.join([go_up(3), name_dir])
    if level_up == 4:
        path = separator.join([go_up(4), name_dir])

    if os.path.exists(path):
        answer = input(f'Path already exists {path}. Do you want to overwrite the files? [y/n] ')
        if answer == 'y':
            # Remove all the files in case they already exist
            shutil.rmtree(path)
            os.makedirs(path)
            logging.info(f"Successfully created the directory '{path}' \n")
        else:
            logging.info(f"Same folder as before {path} \n")
    else:
        os.makedirs(path)