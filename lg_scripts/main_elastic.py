import pystdml as ml
import time

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from elastic_tf_record_dataaset import ElasticTFRecordDataset
from elastic_state import ElasticState, ElasticContext


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


def main_elastic_loop():
    sess = ml.init_elastic()
    print(sess)

    synced = False
    max_step = 1000
    step = 0
    while step < max_step:
        if not synced:
            step = sess.all_reduce_max(step)
            synced = True
        print('step: %d from %s' % (step, sess))

        # BEGIN step work
        time.sleep(1.0 / sess.size())
        # END step work

        result = sess.resize()  # check config server for updates
        if result.changed:
            if result.detached:
                print('%s detached at step %s' % (sess, step))
                break
            synced = False
        step += 1
    print('main_elastic finished')


def main_elastic_state():
    es = ElasticState(100)

    while not es.stopped():
        should_sync = es.begin()
        print('# %d' % (es._progress))
        if should_sync:
            print('user should sync states')

        time.sleep(1.0 / es._sess.size())
        es.end()

    print('main_elastic_state stopped')
    print('stop reason: %s' % (es.stop_reason()))


def main_elastic_context():
    '''
    dataset = create_squad_dataset_2(
        batch_size=1,
        repeat_count=1,
        data_file_path=train_data_file_path,
        schema_file_path=schema_file_path,
        do_shuffle=False,
        device_num=1,
        rank=0,
    )

    it = iter(dataset)
    '''

    es = ElasticState(10)

    while not es.stopped():
        with ElasticContext(es) as should_sync:
            print('# progress %d' % (es._progress))
            if should_sync:
                print('user should sync states')
                #dataset.reset()
                #it = iter(dataset)
                # FIXME: move it to es._progress

            # do step work
            #item = next(it)
            # print(item)
            #for i,t in enumerate(item):
            #    print('{}{}'.format(t.dtype, t.shape))
            #    break
            time.sleep(1.0 / es._sess.size())


# main_elastic_loop()
# main_elastic_state()
main_elastic_context()

#main()
