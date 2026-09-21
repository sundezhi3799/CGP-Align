"""Convert saved training configurations to validated CLI arguments."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def command_from_config(module_name, config):
    module = importlib.import_module(module_name)
    original = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = lambda self, *a, **kw: self
        parser = module.parse_args()
    finally:
        argparse.ArgumentParser.parse_args = original
    argv = []
    recognized = {}
    for action in parser._actions:
        key = action.dest
        if key not in config or not action.option_strings or key == 'help': continue
        value = config[key]
        if isinstance(value, dict) and key in ('dynamic_entity_encoding', 'entity_features'):
            value = value['mode']
        if isinstance(value, (dict, list)):
            raise ValueError('Unexpected structured CLI value: ' + key)
        recognized[key] = value
        if isinstance(action, argparse._StoreTrueAction):
            if value: argv.append(action.option_strings[0])
        elif isinstance(action, argparse._StoreFalseAction):
            if not value: argv.append(action.option_strings[0])
        elif value is not None:
            argv.extend([action.option_strings[0], str(value)])
    parsed = vars(parser.parse_args(argv))
    for key, value in recognized.items():
        if isinstance(parsed[key], Path) and parsed[key] == Path(value): continue
        if str(parsed[key]) != str(value):
            raise ValueError('Config round-trip mismatch: ' + key)
    return [sys.executable, '-u', str(ROOT / 'scripts' / (module_name + '.py'))] + argv
