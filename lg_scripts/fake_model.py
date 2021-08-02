import pystdml as ml
import time
import argparse

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.common.dtype as mstype
from elastic_tf_record_dataaset import ElasticTFRecordDataset
from elastic_state import ElasticState, ElasticContext, ElasticCallback
from mindspore_extension import SleepCallback
from debug_ops import KungFuLogTensor


def parse_args():
    p = argparse.ArgumentParser(description='')
    p.add_argument('--shuffle', action='store_true', default=False)
    return p.parse_args()


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


class QuadraticFunction(ms.nn.Cell):
    """QuadraticFunction is a model that has 1 parameter w and outputs w^2 for any inputs."""
    def __init__(self):
        super(QuadraticFunction, self).__init__()
        init = ms.common.initializer.Normal(0.02)
        shape = [1]

        self.w = ms.common.parameter.Parameter(
            ms.common.initializer.initializer(init, shape),  #
        )

    def construct(self, *args, **kwargs):
        # RuntimeError: mindspore/ccsrc/pipeline/jit/static_analysis/prim.cc:915 GetEvaluatedValueForBuiltinTypeAttrOrMethod] Mindspore don't support this usage '__class__' for the object with type <Tensor[Int32]>.
        # print(args[0].__class__)
        '''
        for a in args:
            print(a)
        for k,v in kwargs.iterms():
            print(k, v)
        '''

        return self.w * self.w


def main_model_train():
    dataset = create_squad_dataset_2(
        batch_size=1,
        repeat_count=1,
        data_file_path=train_data_file_path,
        schema_file_path=schema_file_path,
        do_shuffle=False,
        device_num=1,
        rank=0,
    )
    net = QuadraticFunction()
    model = ms.train.Model(net)

    callbacks = []
    callbacks += [SleepCallback()]

    elastic_state = ElasticState(20)
    elastic_callback = ElasticCallback(elastic_state)
    callbacks += [elastic_callback]

    old_create_tuple_iterator = dataset.__class__.create_tuple_iterator
    print(old_create_tuple_iterator
          )  # <function Dataset.create_tuple_iterator at 0x7f10bfb60830>

    def create_tuple_iterator(self, num_epochs=None, do_copy=None):
        print('create_tuple_iterator intercepted')
        it = old_create_tuple_iterator(self,
                                       num_epochs=num_epochs,
                                       do_copy=do_copy)
        print(
            it
        )  # <mindspore.dataset.engine.iterators.TupleIterator object at 0x7f7eedebd590>
        elastic_callback._iter = it
        return it

    dataset.__class__.create_tuple_iterator = create_tuple_iterator

    #dataset_helper = DatasetHelper(dataset)
    #dataset_helper.iter()
    # iter = TupleIterator(dataset)
    model.train(1, dataset, callbacks=callbacks, dataset_sink_mode=False)


def main_user_loop():
    args = parse_args()

    batch_size = 1024
    dataset = create_squad_dataset_2(
        batch_size=batch_size,
        repeat_count=1,
        data_file_path=train_data_file_path,
        schema_file_path=schema_file_path,
        do_shuffle=args.shuffle,
        device_num=1,
        rank=0,
    )
    net = QuadraticFunction()
    model = ms.train.Model(net)

    callbacks = []
    #callbacks += [SleepCallback()]

    elastic_state = ElasticState(20)
    elastic_callback = ElasticCallback(elastic_state)
    callbacks += [elastic_callback]

    n_epochs = 2

    from mindspore.train.dataset_helper import DatasetHelper
    dataset_helper = DatasetHelper(dataset,
                                   dataset_sink_mode=False,
                                   sink_size=-1,
                                   epoch_num=n_epochs)
    # dataset_helper.iter()

    log_tensor = KungFuLogTensor()
    for epoch in range(n_epochs):
        print('begin epoch %d' % (epoch))
        for step, item in enumerate(dataset_helper):
            print('eopch %4d, step %6d, %d tensors from item' %
                  (epoch, step, len(item)))
            for i, t in enumerate(item):
                print('%d %s%s' % (i, t.dtype, t.shape))
                log_tensor(t)
        '''
        print('will reset dataset')
        dataset.reset()
        print('dataset reset done')
        '''
    print('main_user_loop finished')


print('before main!!!')
# main_elastic_state()
# main_elastic_context()
# main_model_train()
main_user_loop()
