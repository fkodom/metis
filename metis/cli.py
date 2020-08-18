"""
cli.py
------
Provides a command-line interface for `tiki`.
"""

from os import path, getcwd
import argparse
from subprocess import call, DEVNULL
from webbrowser import open_new_tab

from metis import __name__, __version__


local_dir = path.abspath(path.dirname(__file__))
dashboard_file = path.join(local_dir, "dashboard", "dashboard.py")


def main():
    """Provides 2 CLI commands for users:
        * `metis --version` displays the installed version of `metis`
        * `metis dashboard --logdir <path-to-logs>` displays a visualization dashboard

    TODO:
        * `metis docs` displays Sphinx docs for the `metis` API
    """
    parser = argparse.ArgumentParser(description="Get logdir.")
    parser.add_argument("command", default="", nargs="?")
    parser.add_argument("--logdir", dest="logdir", default="logs")
    parser.add_argument("--version", dest="version", action="store_true")
    args = parser.parse_args()

    command = args.command.lower()
    logdir = path.join(getcwd(), args.logdir)
    if command == "dashboard":
        try:
            call(["streamlit", "run", dashboard_file, logdir])
        except KeyboardInterrupt:
            pass
    # TODO: Add Sphinx docs, to enable this CLI option.
    # elif command == "docs":
    #     call([path.join(docs_dir, "make.bat"), "html"], stdout=DEVNULL, stderr=DEVNULL)
    #     open_new_tab(path.realpath(path.join(docs_dir, "docs.html")))

    if args.version:
        print(f"{__name__}, version {__version__}")
