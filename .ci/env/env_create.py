from jinja2 import Template
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "name",
    nargs="?",
    default="scpy.yml",
    help="name of the output yml file (ext must be .yml!) ",
)
parser.add_argument(
    "-v", "--version", default="3.8", help="Python version (default=3.8)"
)
parser.add_argument("--dev", help="make a development environment", action="store_true")
parser.add_argument("--dash", help="use dash", action="store_true")
parser.add_argument("--cantera", help="use cantera", action="store_true")

args = parser.parse_args()

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)

env = Path(__file__).parent
tempfile = env / "env_template.yml"
template = Template(tempfile.read_text("utf-8"))

name = args.name.split(".yml")[0]
out = template.render(
    NAME=name, VERSION=args.version, DEV=args.dev, DASH=args.dash, CANTERA=args.cantera
)

filename = (env / args.name).with_suffix(".yml")
filename.write_text(out)
