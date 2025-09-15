"""
Initializes sftpy module. Creates global runtime configuration in a similar
fashion to matplotlib.
"""

from collections.abc import MutableMapping
from numpy import random as nprand
import re

class SimRC(MutableMapping, dict):

    # TODO rewrite as actual parser

    def __init__(self, sec_unnamed: str=""):

        # unnamed section default for top of file w/o section
        self._sec_unnamed = sec_unnamed

        # pattern for skippable lines
        self._pskip = re.compile(r"^\n|#")
        # pattern for section name
        self._psec = re.compile(r"^\[(\w+(\.\w+)*)\]")
        # pattern for key-value pairs -- this can capture wrong things.
        self._pkvpair = re.compile(r"(\w+)\s*(=|:)\s*(([a-zA-Z0-9._,$](,\s*)?)+)")
        # pattern for ints
        self._pint = re.compile(r"(0(x|b))?(\d+_)*\d+")
        # pattern for floats
        self._pfloat = re.compile(r"((\d+_)*\d+)?\.?\d*(e(\+|-)?\d+)?")
    
    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)


    def rc_from_file(self, fname: str):
        rc = {}
        with open(fname) as file:
            sec = self._sec_unnamed
            for i, line in enumerate(file.readlines(), 1):

                # skip newline or comment
                if self._pskip.match(line) is not None:
                    continue

                # section header
                msec = self._psec.match(line)
                if msec is not None:
                    sec = msec.group(1)
                    continue

                # key-value pair
                mkvpair = self._pkvpair.match(line)

                # if no key-val pair match (after newline/comment and sec), error
                if mkvpair is None:
                    details = (fname, i, 1, line, i, len(line))
                    raise SyntaxError(f"sftpy parse error", details)

                key = sec + "." + mkvpair.group(1)
                val = mkvpair.group(3)

                # int
                mint = self._pint.fullmatch(val)
                #print(mint)
                if mint is not None:
                    if mint.group(2) == "b":
                        val = int(val, 2)
                    elif mint.group(2) == "x":
                        val = int(val, 16)
                    else:
                        val = int(val, 10)
                    rc[key] = val
                    continue

                # float
                mfloat = self._pfloat.fullmatch(val)
                if mfloat is not None:
                    val = float(val)
                    rc[key] = val
                    continue

                # otherwise, the value is assumed to be a string

                # add key-value pair to runtime configuration
                rc[key] = val

        dict.update(self, rc)
        return rc

    def rc_from_files(self, flist: list[str]):
        rc = {}
        for fname in flist:
            rc.update(self.rc_from_file(fname))

        dict.update(self, rc)
        return rc

simrc = SimRC()
simrc.rc_from_files(["sftpy/config/params.ini", "sftpy/config/sft.ini"])
rng = nprand.default_rng(seed=simrc["general.seed"])

