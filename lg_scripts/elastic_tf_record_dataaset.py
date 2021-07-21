import mindspore._c_dataengine as cde
from mindspore.dataset import (Schema, Shuffle, SourceDataset,
                               check_tfrecorddataset, replace_none)


# modified from TFRecordDataset
class ElasticTFRecordDataset(SourceDataset):
    """
    A source dataset for reading and parsing datasets stored on disk in TFData format.

    The columns of generated dataset depend on the source TFRecord files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in a lexicographical order.
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the meta data from the TFData file is considered the schema.
        columns_list (list[str], optional): List of columns to be read (default=None, read all columns).
        num_samples (int, optional): The number of samples (rows) to be included in the dataset (default=None).
            If num_samples is None and numRows(parsed from schema) does not exist, read the full dataset;
            If num_samples is None and numRows(parsed from schema) is greater than 0, read numRows rows;
            If both num_samples and numRows(parsed from schema) are greater than 0, read num_samples rows.
        num_parallel_workers (int, optional): Number of workers to read the data
            (default=None, number set in the config).
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into (default=None). When this argument is specified, `num_samples` reflects
            the maximum sample number of per shard.
        shard_id (int, optional): The shard ID within num_shards (default=None). This
            argument can only be specified when num_shards is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards(default=False). If shard_equal_rows
            is false, number of rows of each shard may be not equal, and may lead to a failure in distributed training.
            When the number of samples of per TFRecord file are not equal, it is suggested to set to true.
            This argument should only be specified when num_shards is also specified.
        cache (DatasetCache, optional): Use tensor caching service to speed up dataset processing.
            (default=None, which means no cache is used).

    Raises:
        RuntimeError: If dataset_files are not valid or do not exist.
        RuntimeError: If num_parallel_workers exceeds the max thread numbers.
        RuntimeError: If num_shards is specified but shard_id is None.
        RuntimeError: If shard_id is specified but num_shards is None.
        ValueError: If shard_id is invalid (< 0 or >= num_shards).

    Examples:
        >>> import mindspore.common.dtype as mstype
        >>>
        >>> tfrecord_dataset_dir = ["/path/to/tfrecord_dataset_file"] # contains 1 or multiple TFRecord files
        >>> tfrecord_schema_file = "/path/to/tfrecord_schema_file"
        >>>
        >>> # 1) Get all rows from tfrecord_dataset_dir with no explicit schema.
        >>> # The meta-data in the first row will be used as a schema.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir)
        >>>
        >>> # 2) Get all rows from tfrecord_dataset_dir with user-defined schema.
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col_1d', de_type=mstype.int64, shape=[2])
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=schema)
        >>>
        >>> # 3) Get all rows from tfrecord_dataset_dir with schema file.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=tfrecord_schema_file)
    """
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
        return cde.ElasticTFRecordNode(self.dataset_files, schema,
                                       self.columns_list, self.num_samples,
                                       self.shuffle_flag, self.num_shards,
                                       self.shard_id, self.shard_equal_rows)
