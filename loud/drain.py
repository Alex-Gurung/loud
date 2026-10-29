"""Wind GPU work down without losing any: with the drain on, every worker finishes the job it is running and exits
instead of claiming another (queued jobs stay queued). Turn it off before launching workers again.

    python3 -m loud.drain on        # workers exit as their current jobs finish
    python3 -m loud.drain off       # workers claim again (new pods exit at once while it is on)
    python3 -m loud.drain           # say whether it is on
"""
import argparse
from pathlib import Path

from .paths import QUEUE

FILE = 'drain'


def is_on(queue=QUEUE):
    return (Path(queue) / FILE).exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('state', nargs='?', choices=('on', 'off'))
    parser.add_argument('--queue', type=Path, default=QUEUE)
    args = parser.parse_args()
    path = args.queue / FILE
    if args.state == 'on':
        path.write_text('workers finish their current jobs and exit\n')
    elif args.state == 'off' and path.exists():
        path.unlink()
    print(f'drain is {"on: workers exit after their current jobs" if path.exists() else "off"} ({args.queue})')


if __name__ == '__main__':
    main()
