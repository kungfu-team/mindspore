# from src.dataset import create_squad_dataset
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from elastic_tf_record_dataaset import ElasticTFRecordDataset


def create_squad_dataset_2(
    batch_size=1,
    repeat_count=1,
    data_file_path=None,
    schema_file_path=None,
    do_shuffle=True,
    device_num=1,
    rank=0,
):
    type_cast_op = C.TypeCast(mstype.int32)
    # TFRecordDataset < SourceDataset < Dataset
    data_set = ElasticTFRecordDataset(
        [
            data_file_path,
        ],
        schema_file_path,
        columns_list=[
            "input_ids",
            "input_mask",
            "segment_ids",
            "start_positions",
            "end_positions",
            "unique_ids",
            "is_impossible",
        ],
        shuffle=do_shuffle,
        num_shards=device_num,
        shard_id=rank,
        shard_equal_rows=True,
    )
    data_set = data_set.map(operations=type_cast_op,
                            input_columns="start_positions")
    data_set = data_set.map(operations=type_cast_op,
                            input_columns="end_positions")

    data_set = data_set.map(operations=type_cast_op,
                            input_columns="segment_ids")
    data_set = data_set.map(operations=type_cast_op,
                            input_columns="input_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")

    data_set = data_set.map(operations=type_cast_op,
                            input_columns="unique_ids")
    # mindspore.dataset.engine.datasets.MapDataset < Dataset
    print(data_set.__class__)

    data_set = data_set.repeat(repeat_count)
    # mindspore.dataset.engine.datasets.RepeatDataset < Dataset
    print(data_set.__class__)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    # mindspore.dataset.engine.datasets.BatchDataset < Dataset
    print(data_set.__class__)

    return data_set


train_data_file_path = "/data/squad1/train.tf_record"
schema_file_path = "/data/squad1/squad_schema.json"

dataset = create_squad_dataset_2(
    batch_size=1,
    repeat_count=1,
    data_file_path=train_data_file_path,
    schema_file_path=schema_file_path,
    do_shuffle=False,
    device_num=1,
    rank=0,
)

n = dataset.get_dataset_size()  # == 88641
print(n)  # 88641
'''
tot = 0
for i, items in enumerate(iter(dataset)):
    if (i + 1) % 10000 == 0:
        print('{} {}'.format(i, items))
    tot += 1

print('total: {}'.format(tot))
'''

dataset = dataset.batch(5)
dataset = dataset.batch(5)
n = dataset.get_dataset_size()
print(n)

for i, items in enumerate(dataset):
    print('# %d' %(i))
    for t in items:
        print('{}{}'.format(t.dtype, t.shape))
    # break

'''
 $ cat err.log | grep 'ElasticTFReaderOp::LoadFeature(?, ?, ?, col=6)'| wc -l
88641
'''


