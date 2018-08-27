import model
from mxnet.gluon import nn
import argparse
from mxnet import image
from mxnet import nd
import mxnet as mx
from mxnet.contrib.ndarray import MultiBoxDetection
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--params", help="name of params", type=str, default='DSOD.params')
parser.add_argument("--size", help="resolution of image", type=int, default=512)
parser.add_argument("--gpu",help="use GPU or not", type=int, default=1)
parser.add_argument("--image",help="inference image ", type=str, default='pikachu.jpg')
parser.add_argument("--score",help="displayed when box score greater than this threshold ", type=float, default=0.5)
args = parser.parse_args()


rgb_mean = nd.array([123, 117, 104])
data_shape = args.size
if args.gpu:
    ctx = mx.gpu(0)
else:
    ctx = mx.cpu(0)

def process_image(fname):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())
    # resize to data_shape
    data = image.imresize(im, data_shape, data_shape)
    # minus rgb mean
    data = data.astype('float32') - rgb_mean
    # convert to batch x channel x height xwidth
    return data.transpose((2, 0, 1)).expand_dims(axis=0), im


def predict(x):
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))

    cls_probs = nd.SoftmaxActivation(
        cls_preds.transpose((0, 2, 1)), mode='channel')

    return MultiBoxDetection(cls_probs, box_preds, anchors,
                             force_suppress=True, clip=False, nms_threshold=0.1)  # ,nms_threshold=0.1


def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        fill=False, edgecolor=color, linewidth=linewidth)


def display(im, out, threshold=0.5):

    colors = ['blue', 'green', 'red', 'black', 'magenta']
    plt.imshow(im.asnumpy())
    for row in out:
        row = row.asnumpy()
        class_id, score = int(row[0]), row[1]
        if class_id < 0 or score < threshold:
            continue
        color = colors[class_id % len(colors)]
        box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
        rect = box_to_rect(nd.array(box), color, 2)
        plt.gca().add_patch(rect)

        #
        text = 'pikachu'
        plt.gca().text(box[0], box[1],
                       '{:s} {:.2f}'.format(text, score),
                       bbox=dict(facecolor=color, alpha=0.5),
                       fontsize=10, color='white')
    # print(time.time() - tic)

    plt.show()




if __name__ == '__main__':
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            model.DSOD(32, 6, 32, 1, 1)
        )

    net.load_params(args.params, ctx=mx.gpu())
    x, im = process_image(args.image)
    out = predict(x)
    display(im, out[0], threshold=args.score)


