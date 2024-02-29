import os
import shutil
import subprocess as sp
import glob


def test_all():
    script_list = [
        'classification.py',
        'image_regression.py',
    ]
    for f in script_list:
        command = 'python {}'.format(os.path.join(os.pardir, 'examples', f))
        child = sp.Popen(command.split(' '), stdout=sp.PIPE)
        _ = child.communicate()[0]
        rc = child.returncode
        assert rc == 0, '{} returned with non-zero status.'.format(f)
    shutil.rmtree('temp')
    return


if __name__ == '__main__':
    test_all()
