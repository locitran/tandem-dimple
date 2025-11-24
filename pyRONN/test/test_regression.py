#!/usr/bin/env python

# Python 2 and 3 compatibility
from __future__ import print_function

from itertools import islice

import numpy as np
import pandas as pd
import Bio.SeqIO

import ronn

def raw_score_compare(old_output_fn, records_fn):
    """
    """
    records = Bio.SeqIO.parse(records_fn, 'fasta')
    old_output = open(old_output_fn, 'r+')

    # first sequence record
    currecord = next(records, False)
    while(currecord):
        # read the corresponding number of rows from the reference output file
        # +1 to account for the header/comment line
        old = pd.DataFrame(list(islice(old_output, len(currecord.seq)+1))[1:])

        old = old[0].str.split(expand=True)
        try:
            old.columns = ['RESIDUE', 'SCORES']
        except ValueError:
            print(old)
            raise

        # calculation on new sequence
        try:
            new = ronn.calc_ronn(str(currecord.seq))
        except TypeError as err:
            print(currecord.seq[10:])
            raise

        # check that a score was returned for each position
        if len(currecord.seq) != len(new):
            print(currecord.seq[10:], len(currecord.seq))
            print(new[10:], len(new))
            raise AssertionError("Score seems missing for a position")

        # check that scores match
        # `atol` becuase of the precision to which `old` was printed
        if not np.allclose(pd.to_numeric(old['SCORES']),
                           new, atol=1e-04, rtol=1):
            print("OLD\n", old.tail())
            print("NEW\n", new.tail(), flush=True)
            raise AssertionError("`old` and `new` don't match for %s " %
                                 currecord.seq[10:])

        currecord = next(records, False)

    return


def test_ecolik12():
    raw_score_compare("test/Daley_gfp.short.ronn32", "test/Daley_gfp.short.fna.faa")
    return


def test_test1():
    raw_score_compare("test/test1.ronn32", "test/test1.fna.faa")
    return
