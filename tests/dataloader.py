import unittest
from dataloader.DataLoader import DataLoader

datafolder = 'tests/mock_data/EMG_dataset/'
resample_fs = None


class DataloaderTestCase(unittest.TestCase):
    
    def test_get_paths_training_set(self):
        loader_class = DataLoader(datafolder, True, resample_fs)
        
        file_paths = loader_class.__get_paths__()
        expected = [datafolder+'01/2_raw_data_13-13_22.03.16.txt',
                    datafolder+'02/2_raw_data_14-21_22.03.16.txt',
                    datafolder+'03/2_raw_data_09-34_11.04.16.txt']
        self.assertEqual(file_paths,expected,'Path of text files with prefix 2_ should be loaded')
        self.assertEqual(len(file_paths), 3, 'It should return a total of 3 file paths')
        self.assertNotIn('mock_data/EMG_dataset/README.txt', file_paths,
                         'It should not load any extraneous path')

    def test_get_paths_test_set(self):
        loader_class = DataLoader(datafolder, False, resample_fs)
        
        file_paths = loader_class.__get_paths__()
        expected = [datafolder+'01/1_raw_data_13-12_22.03.16.txt',
                    datafolder+'02/1_raw_data_14-19_22.03.16.txt',
                    datafolder+'03/1_raw_data_09-32_11.04.16.txt']
        self.assertEqual(file_paths,expected,'Path of text files with prefix 1_ should be loaded')
        self.assertEqual(len(file_paths), 3, 'It should return a total of 3 file paths')
        self.assertNotIn('mock_data/EMG_dataset/README.txt', file_paths,
                         'It should not load any extraneous path')
        
if __name__ == '__main__':
    unittest.main()