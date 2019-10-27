import sys
sys.path.append(r"D:\GitHub\studyNote\python基础\python_dir\app\sub2")
import mod2
def print_mod1():
    print('__name__: {}'.format(__name__))
    print('__package__: {}'.format(__package__))
    print('Import Successfully!')
    mod2.print_mod2()

