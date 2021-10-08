__authors__ = ["J. Kieffer"]
__license__ = "MIT"
__date__ = "08/10/2021"

import numpy
import unittest
from ..parser import topas_parser
import logging
logger = logging.getLogger(__name__)
import time


class TestParse(unittest.TestCase):

    def test_parse_topas(self):
        pass


def suite():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestParse("test_parse_topas"))
    return testSuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
