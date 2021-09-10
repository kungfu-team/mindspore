import mindspore._c_dataengine as cde
from mindspore.dataset import (Schema, Shuffle, SourceDataset,
                               check_tfrecorddataset, replace_none)


# modified from TFRecordDataset
class ElasticTFRecordDataset(SourceDataset):
    @check_tfrecorddataset
    def __init__(self,
                 dataset_files,
                 schema=None,
                 columns_list=None,
                 num_samples=None,
                 num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL,
                 num_shards=None,
                 shard_id=None,
                 shard_equal_rows=False,
                 cache=None):
        if True:
            print('creating ElasticTFRecordDataset python class')
        super().__init__(num_parallel_workers=num_parallel_workers,
                         num_samples=num_samples,
                         shuffle=shuffle,
                         num_shards=num_shards,
                         shard_id=shard_id,
                         cache=cache)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

        self.schema = schema
        self.columns_list = replace_none(columns_list, [])
        self.shard_equal_rows = replace_none(shard_equal_rows, False)

        if self.schema is not None and (self.num_samples is None
                                        or self.num_samples == 0):
            self.num_samples = Schema.get_num_rows(self.schema)

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema,
                                                      Schema) else self.schema
        if True:
            print('creating cde.ElasticTFRecordNode python class')
        return cde.ElasticTFRecordNode(self.dataset_files, schema,
                                       self.columns_list, self.num_samples,
                                       self.shuffle_flag, self.num_shards,
                                       self.shard_id, self.shard_equal_rows)

    def sync_state(self):
        pass
        print('TODO: sync state')
