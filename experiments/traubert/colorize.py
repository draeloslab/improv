#! /usr/bin/env python

import argparse
from pathlib import Path
import re
import sys

import colorama

DATETIME_RE = re.compile(r'20\d\d-\d\d-\d\d \d\d:\d\d:\d\d(\.\d*)?')
DIGIT_RE = re.compile(r'\d')

green_line_patterns = [
    re.compile(r'jdg', flags=re.IGNORECASE),
    re.compile(r'improv.nexus Allowing start'),
    re.compile(r'error', flags=re.IGNORECASE),
]

def normalize(line):
    line = DATETIME_RE.sub('DATETIME', line)
    line = DIGIT_RE.sub('X', line)
    return line

def load_seen(path):
    seen = set()
    if path.is_file():
        with path.open('r') as f:
            for line in f:
                seen.add(normalize(line))
    return seen

def main(old_log_path=Path('global.old.log'), filter_uninteresting=False):
    seen = load_seen(old_log_path)

    for line in sys.stdin:
        if normalize(line) in seen:
            for pattern in green_line_patterns:
                if re.search(pattern, line) is not None:
                    sys.stdout.write(colorama.Fore.GREEN + line + colorama.Style.RESET_ALL)
                    break
            else:
                if not filter_uninteresting:
                    # sys.stdout.write(colorama.Fore.BLACK + line + colorama.Style.RESET_ALL)
                    sys.stdout.write(line)
        else:
            sys.stdout.write(colorama.Fore.RED + line + colorama.Style.RESET_ALL)
        sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize streamed log lines.')
    parser.add_argument(
        '-o', '--old-log',
        type=Path,
        default=Path('global.old.log'),
        help='baseline log used to classify lines as seen or new',
    )
    parser.add_argument(
        '-u', '--filter-uninteresting',
        action='store_true',
        help='omit seen lines that do not match the green-line patterns',
    )
    args = parser.parse_args()
    main(args.old_log, args.filter_uninteresting)
