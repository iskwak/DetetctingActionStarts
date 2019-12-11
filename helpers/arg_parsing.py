"""Helper to deal with arguments."""
from __future__ import print_function, division
import os
import gflags
import numpy
import sys

gflags.DEFINE_boolean("help", False, "Print this message.")
gflags.DEFINE_boolean("debug", False, "General debug flag.")


def check_help(flags):
    """See if it printing help is needed."""
    if flags.help is True:
        print(flags)
        return True
    return False


def copy_flags(module_name, gflags, opts=None):
    """Convert the gflags object to a dictionary."""
    # This just makes things a tad bit easier to manipulate...
    all_modules = gflags.FlagsByModuleDict()
    module_flags = all_modules[module_name]

    if opts is None:
        opts = {}

    for flag in module_flags:
        opts[flag.name] = flag.value

    return opts


def copy_all_flags(gflags, opts=None):
    """Convert all gflags into a dictionary of dictionaries."""
    if opts is None:
        opts = dict()

    all_modules = gflags.FlagsByModuleDict()
    module_names = list(all_modules.keys())
    # loop over each of the modules.
    for module_name in module_names:
        key_name = shorten_module_name(module_name)
        opts[key_name] = copy_flags(module_name, gflags)

    return opts


def setup_opts(argv, flags):
    """Does this work?"""
    opts = dict()

    opts["eps"] = numpy.finfo(numpy.float32).eps
    opts["argv"] = argv

    flags(argv)
    if check_help(flags) is True:
        sys.exit()

    if flags.debug is True:
        opts["rng"] = numpy.random.RandomState(123)
    else:
        opts["rng"] = numpy.random.RandomState()

    # is this needed?
    opts["flags"] = flags

    # opts = copy_all_flags(flags, opts=opts)

    return opts


def shorten_module_name(module_name):
    """The module name may have path information in it, remove this."""
    (path, base_name) = os.path.split(module_name)
    split_str = base_name.split(".")

    if split_str[-1] == "py":
        return split_str[-2]
    else:
        return split_str[-1]
