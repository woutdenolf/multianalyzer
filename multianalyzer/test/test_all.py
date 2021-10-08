#!/usr/bin/env python
# coding: utf-8

__author__ = "Guillaume"
__license__ = "MIT"
__copyright__ = "2015, ESRF"
__date__ = "08/10/2021"

import unittest
from . import test_parser


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_parser.suite())
    return testSuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
