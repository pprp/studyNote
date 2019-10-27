import sys

sys.path.append(r"D:\GitHub\studyNote\python基础\python_dir\app\sub1")

import mod1
def print_mod2():
    print('__name__: {}'.format(__name__))
    print('__package__: {}'.format(__package__))
    print('Import Successfully!')

mod1.print_mod1()