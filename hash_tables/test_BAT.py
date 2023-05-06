import unittest
from oohtable import HashTable
class TestHashTable(unittest.TestCase):
    def test_init(self):
        # test the initial values
        ht = HashTable(10)
        self.assertEqual(len(ht), 0)
        self.assertEqual(ht.buckets, [[] for i in range(10)])

    def test_setitem(self):
        # test setting values
        ht = HashTable(10)
        ht['key'] = 'value'
        self.assertEqual(len(ht), 1)
        self.assertEqual(ht['key'], 'value')
        ht['key'] = 'new value'
        self.assertEqual(len(ht), 1)
        self.assertEqual(ht['key'], 'new value')

    def test_getitem(self):
        # test getting values
        ht = HashTable(10)
        ht['key'] = 'value'
        self.assertEqual(ht['key'], 'value')
        self.assertIsNone(ht['non-existing key'])

    def test_contains(self):
        # test the contains operator
        ht = HashTable(10)
        ht['key'] = 'value'
        self.assertTrue('key' in ht)
        self.assertFalse('non-existing key' in ht)

    def test_iter(self):
        # test iteration
        ht = HashTable(10)
        ht['key1'] = 'value1'
        ht['key2'] = 'value2'
        ht['key3'] = 'value3'
        keys = []
        for key in ht:
            keys.append(key)
        self.assertEqual(sorted(keys), ['key1', 'key2', 'key3'])

    def test_keys(self):
        # test getting all keys
        ht = HashTable(10)
        print(ht.__repr__())
        ht['key1'] = 'value1'
        ht['key2'] = 'value2'
        ht['key3'] = 'value3'
        self.assertEqual(sorted(ht.keys()), ['key1', 'key2', 'key3'])


if __name__ == '__main__':
    unittest.main()