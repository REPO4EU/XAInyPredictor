import unittest
from typing import ClassVar

from mymodule.template import TemplateClass


class TestMyModule(unittest.TestCase):
    test: ClassVar[TemplateClass]

    @classmethod
    def setUpClass(self):
        self.test = TemplateClass("./test/data/testInput.tsv", threshold=10)

    def test_square(self):
        self.assertEqual(self.test._square(10), 100)


if __name__ == "__main__":
    unittest.main()
