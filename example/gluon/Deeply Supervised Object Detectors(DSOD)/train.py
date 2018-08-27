import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import model
from mxnet import metric
import time
import os.path as osp
from mxnet.contrib.ndarray import MultiBoxTarget
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch", help="number of batch", type=int, default=4)
parser.add_argument("--size", help="resolution of image", type=int, default=512)
parser.add_argument("--epoch", help="number of epochs", type=int, default=0)
parser.add_argument("--retrain", help="load weights and continue training", type=int, default=0)
parser.add_argument("--dataset",help="name of .rec and .lst ( must the same name)", type=str, default='data/train')
parser.add_argument("--params",help="name of params", type=str, default='DSOD.params')
parser.add_argument("--gpu",help="use GPU or not", type=int, default=1)
args = parser.parse_args()

def training_targets(anchors, class_preds, labels):
    class_preds = class_preds.transpose(axes=(0, 2, 1))

    return MultiBoxTarget(anchors, labels, class_preds,
                          overlap_threshold=0.3)  # ,overlap_threshold=0.3,negative_mining_ratio=0.3

class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)

        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = - self._alpha * ((1 - pj) ** self._gamma) * pj.log()

        return loss.mean(axis=self._batch_axis, exclude=True)

class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)

if args.gpu:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu(0)

batch_size = args.batch
data_shape = args.size

def get_iterators(data_shape, batch_size):
    file_exists = osp.exists('data/train.rec') and osp.exists('data/train.idx')
    if not file_exists:
        import getPikachu
        getPikachu.get_Dataset()
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = mx.image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=args.dataset +'.rec',
        path_imgidx=args.dataset +'.idx',
        shuffle=True,
        mean=True
    )
    return train_iter, class_names

def train(train_iter):

    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            model.DSOD(32, 6, 32, 1, 1)  # 64 6 48 1 1
        )
    net.initialize()

    box_loss = SmoothL1Loss()
    cls_loss = FocalLoss()  # hard neg mining vs FocalLoss()
    l1_loss = gluon.loss.L1Loss()
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': 0.1, 'wd': 5e-4})

    cls_metric = metric.Accuracy()
    box_metric = metric.MAE()

    filename = args.params
    if args.retrain:
        print('load last time weighting')
        net.load_params(filename, ctx=mx.gpu())

    for epoch in range(args.epoch):
        train_data.reset()
        cls_metric.reset()
        box_metric.reset()
        tic = time.time()

        for i, batch in enumerate(train_data):
            x = batch.data[0].as_in_context(ctx)
            y = batch.label[0].as_in_context(ctx)

            with mx.autograd.record():
                anchors, class_preds, box_preds = net(x)
                box_target, box_mask, cls_target = training_targets(anchors, class_preds, y)

                loss1 = cls_loss(class_preds, cls_target)

                loss2 = l1_loss(box_preds, box_target, box_mask)

                loss = loss1 + 5 * loss2
            loss.backward()
            trainer.step(batch_size)

            cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
            box_metric.update([box_target], [box_preds * box_mask])

        print('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' % (
            epoch, *cls_metric.get(), *box_metric.get(), time.time() - tic))

        net.save_params(filename)

if __name__ == '__main__':
    train_data, class_names = get_iterators(data_shape, batch_size)
    train_data.reshape(label_shape=(3, 5))
    train(train_data)


