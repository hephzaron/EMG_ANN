import unittest

from tests.dataloader import DataloaderTestCase

scale_test_suite = unittest.TestSuite([
    unittest.TestLoader().loadTestsFromTestCase(DataloaderTestCase)
    ])

def test_scale_suite():
    result = unittest.TestResult()
    runner = unittest.TextTestRunner()
    print(runner.run(scale_test_suite))
    
if __name__ == '__main__':
    test_scale_suite()