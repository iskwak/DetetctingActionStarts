"""Wrapper for testing analyze_outputs2."""
import helpers.analyze_outputs2 as analyze_outputs2

arg_string = [
    'feedforward.py',
    '/groups/branson/bransonlab/kwaki/test_data/data/real',
    '/groups/branson/bransonlab/kwaki/test_data'
]

analyze_outputs2.main(arg_string)
