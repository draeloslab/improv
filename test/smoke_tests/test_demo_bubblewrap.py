from subprocess import Popen, PIPE, run
import time
import re
from pathlib import Path
import os

def show_output_for_time(p, max_seconds=5., continue_regex=None, error_on_timeout=True, expect_exit=False):
    start_t = time.time()
    while time.time() - start_t < max_seconds:
        if p.poll() is not None:
            if expect_exit:
                return
            else:
                raise Exception('Process exited unexpectedly.')
        line = p.stdout.readline()
        if not line:
            if expect_exit:
                return
            else:
                raise Exception('Process exited unexpectedly.')
        print(line, end='')
        if continue_regex is not None and re.search(continue_regex, line) is not None:
            return

        if error_on_timeout and continue_regex is not None:
            raise Exception('Timeout error.')


def test_bubblewrap():
    cwd = Path.cwd()
    assert cwd.name == 'improv', 'we should be in the improv directory'
    assert cwd.exists()
    assert (cwd / 'improv').exists()

    if not (cwd / 'demos' / 'bubblewrap' / 'data' / 'indy_20160407_02.mat').exists():
        print('Downloading data.')
        run(['python', 'demos/bubblewrap/actors/utils.py'])

    result = run(['improv', 'cleanup'], capture_output=True, text=True, timeout=1)
    assert re.search(r'No running processes found\.', result.stdout), 'Environment needs to be clean.'


    open('global.log', 'w').close()

    env = os.environ.copy()
    size = os.get_terminal_size()
    env['COLUMNS'] = str(size.columns)
    env['LINES'] = str(size.lines)
    p = Popen(
        [
            'improv',
            'run',
            '/home/jgould/Documents/naumann_collab/improv/demos/bubblewrap/bubble_demo.yaml'
        ],
        stdout=PIPE,
        stdin=PIPE,
        text=True,
        env=env
    )

    show_output_for_time(p, 15, r'improv console', error_on_timeout=False)

    p.stdin.write('setup\r')
    p.stdin.flush()

    show_output_for_time(p, 10, r'improv\.nexus Allowing start', error_on_timeout=False)

    p.stdin.write('run\r')
    p.stdin.flush()

    show_output_for_time(p, 10)

    p.stdin.write('stop\r')
    p.stdin.flush()

    show_output_for_time(p, .1)

    p.stdin.write('quit\r')
    p.stdin.flush()

    show_output_for_time(p, 2, expect_exit=True)

    if p.poll() is None:
        p.kill()
        raise Exception('Process did not exit when expected.')


if __name__ == '__main__':
    test_bubblewrap()