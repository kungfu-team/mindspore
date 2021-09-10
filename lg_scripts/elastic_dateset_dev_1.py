import os
import time
import argparse

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from elastic_tf_record_dataaset import ElasticTFRecordDataset
# from elastic_state import ElasticCallback
from kungfu.python.elastic_state import ElasticState, ElasticContext
from mindspore_extension import SleepCallback
from debug_ops import KungFuLogTensor
from kungfu.python import current_rank, current_cluster_size, propose_new_size


def parse_args():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--shuffle', action='store_true', default=False)
    p.add_argument('--reload', action='store_true', default=False)
    p.add_argument('--run', action='store_true', default=False)
    p.add_argument('--max-progress', type=int, default=10)
    p.add_argument('--global-batch-size', type=int, default=1)

    return p.parse_args()


def create_squad_dataset_2(
    init_progress=0,
    batch_size=None,
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
    sync_fn = data_set.sync_state

    data_set = data_set.skip(init_progress)

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
    # print(data_set.__class__)

    # data_set = data_set.repeat(repeat_count)
    # mindspore.dataset.engine.datasets.RepeatDataset < Dataset
    # print(data_set.__class__)

    # apply batch operations
    if batch_size:
        data_set = data_set.batch(batch_size, drop_remainder=True)
    # mindspore.dataset.engine.datasets.BatchDataset < Dataset
    # print(data_set.__class__)

    return data_set, sync_fn


train_data_file_path = "/data/squad1/train.tf_record"
schema_file_path = "/data/squad1/squad_schema.json"


def ckpt(es):
    return 'progress-%010d.log' % (es._progress)


def read_step(es):
    with open(ckpt(es)) as f:
        return int(f.read().strip())


def save_step(es, step):
    with open(ckpt(es), 'w') as f:
        f.write('%d\n' % (step))


def main_elastic_context():
    args = parse_args()
    global_batch_size = args.global_batch_size
    batch_size = None

    # for idx, elem in enumerate(dataset):
    #     if idx % 1000 != 0: continue
    #     print('%d : Tuple of %d' % (idx, len(elem)))
    #     for i, t in enumerate(elem):
    #         print('  - [%d] : %s%s' % (i, t.dtype, t.shape))

    #if idx > 2: break

    #time.sleep(3.5)
    # print('done')
    # return
    log_tensor = KungFuLogTensor()

    es = ElasticState(args.max_progress, args.reload)
    progress = es._progress

    rank = current_rank()
    size = current_cluster_size()
    print('%d/%d, starting from %d' % (rank, size, progress))

    # TODO: sync dataset to progress

    if progress == 0 and rank == 0:
        propose_new_size(size)  # write identical config to config server

    dataset, sync_ds = create_squad_dataset_2(
        init_progress=progress,
        batch_size=batch_size,
        repeat_count=1,
        data_file_path=train_data_file_path,
        schema_file_path=schema_file_path,
        do_shuffle=False,
        device_num=1,
        rank=0,
    )

    it = iter(dataset)

    if not args.run:
        return

    step = 0
    if rank == 0:
        if progress > 0:
            step = read_step(es)
            print('init step=%d' % (step))

    while not es.stopped():
        delta = global_batch_size
        with ElasticContext(es, delta) as should_sync:
            # print('# progress %d' % (es._progress))
            if should_sync:
                # TODO: in reload mode, is sync required?
                # print('user should sync states')
                # sync_ds()
                #dataset.reset()
                #it = iter(dataset)
                # FIXME: move it to es._progress
                pass

            step += 1
            progress = es._progress
            if rank == 0:
                # print('progress: %d/%s' % (progress, args.max_progress))
                pass

            # do step work
            item = next(it)

            for i, t in enumerate(item):
                # print('[{}] {}{}'.format(i, t.dtype, t.shape))
                if rank == 0:
                    # log_tensor(t)
                    pass

                # print(t.__class__)
                # for d in dir(t):
                #     print(d)
                # tt = t.to_tensor()
                # print(tt.__class__)
                # x = t.asnumpy()
                # print('[{}] {}{}'.format(i, x.dtype, x.shape))
                break
            # time.sleep(1.0 / es._sess.size())

            if rank == 0:
                period = 300
                if step % period == 0:
                    new_size = (step // period) % 4 + 1
                    propose_new_size(new_size)

    progress = es._progress

    if rank == 0:
        save_step(es, step)


# print('before main!!!')


def log_env():
    for k in os.environ:
        if k.startswith('KUNGFU'):
            v = os.getenv(k)
            print('%s=%s' % (k, v))


def main():
    ms.context.set_context(
        mode=ms.context.GRAPH_MODE,
        #    device_target=args.device,
        save_graphs=False)
    main_elastic_context()


# log_env()
main()
