"""
A HashTable represented as a list of lists with open hashing.
Each bucket is a list of (key,value) tuples
"""

class HashTable:
    def __init__(self, nbuckets):
        """Init with a list of nbuckets lists"""
        # pass
        self.buckets = [[] for _ in range(nbuckets)]


    def __len__(self):
        """
        number of keys in the hashable
        """
        return sum(len(bucket) for bucket in self.buckets)


    def __setitem__(self, key, value):
        """
        Perform the equivalent of table[key] = value
        Find the appropriate bucket indicated by key and then append (key,value)
        to that bucket if the (key,value) pair doesn't exist yet in that bucket.
        If the bucket for key already has a (key,value) pair with that key,
        then replace the tuple with the new (key,value).
        Make sure that you are only adding (key,value) associations to the buckets.
        The type(value) can be anything. Could be a set, list, number, string, anything!
        """
        h = hash(key) % len(self.buckets)
        #pass
        for i, kv in enumerate(self.buckets[h]):
            if kv[0] == key:
                self.buckets[h][i] = (key, value)
                return
        self.buckets[h].append((key, value))
 


    def __getitem__(self, key):
        """
        Return the equivalent of table[key].
        Find the appropriate bucket indicated by the key and look for the
        association with the key. Return the value (not the key and not
        the association!). Return None if key not found.
        """
        h = hash(key) % len(self.buckets)
        # pass
        for k, v in self.buckets[h]:
            if k == key:
                return v
        return None



    def __contains__(self, key):
        # pass
        h = hash(key) % len(self.buckets)
        for k, v in self.buckets[h]:
            if k == key:
                return True
        return False
        


    def __iter__(self):
        # pass
        for bucket in self.buckets:
            for key, value in bucket:
                yield key


    def keys(self):
        """
        return all keys in the hashtable

        Returns
        -------
        elems : TYPE
            DESCRIPTION.

        """
        # pass
        return [key for bucket in self.buckets for key, value in bucket]



    def items(self):
        """
        returns all values in the hashable

        """
        # pass
        return [(key, value) for bucket in self.buckets for key, value in bucket]



    def __repr__(self):
        """
        Return a string representing the various buckets of this table.
        The output looks like:
            0000->
            0001->
            0002->
            0003->parrt:99
            0004->
        where parrt:99 indicates an association of (parrt,99) in bucket 3.
        """
        # pass
        repr_str = ""
        for i, bucket in enumerate(self.buckets):
            repr_str += f"{i:04d}->"
            for key, value in bucket:
                repr_str += f"{key}:{value}->"
            repr_str.strip("->")
            repr_str += "\n"
        return repr_str


    def __str__(self):
        """
        Return what str(table) would return for a regular Python dict
        such as {parrt:99}. The order should be in bucket order and then
        insertion order within each bucket. The insertion order is
        guaranteed when you append to the buckets in htable_put().
        """
        #pass
        str_ = "{"
        for i, bucket in enumerate(self.buckets):
            if bucket:
                bucket_str = ', '.join([str(key) + ':' + str(value) for key,value in bucket])
                str_ += bucket_str + ', '
                
        str_ = str_.strip(', ')
        return str_ + '}'



    def bucket_indexof(self, key):
        """
        You don't have to implement this, but I found it to be a handy function.

        Return the index of the element within a specific bucket; the bucket is:
        table[hashcode(key) % len(table)]. You have to linearly
        search the bucket to find the tuple containing key.
        """
        h = hash(key) % len(self.buckets)
        # # pass
        for i, (k, v) in enumerate(self.buckets[h]):
            if key == k:
                return i
        return None