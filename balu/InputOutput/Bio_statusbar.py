# -*- coding: utf-8 -*-
import numpy as np
import sys


def Bio_statusbar(*args):
    if len(args) == 1:
        s = args[0]
        if s is str:
            return s
        else:
            return 'Progress ...'
    elif len(args) == 2:
        s = args[0]
        if s is str:
            return s
        else:
            v = args[0]
            s = args[1]
            point = int(20*v)
            sys.stdout.write(
                "\r[" + "=" * point + " " * (20 - point) + "] " + str(int(v * 100)) + "% ")
            sys.stdout.flush()
