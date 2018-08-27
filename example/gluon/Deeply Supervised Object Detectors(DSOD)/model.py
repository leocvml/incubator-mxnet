from mxnet import nd
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxPrior

class stemblock(nn.HybridBlock):
    def __init__(self, filters):
        super(stemblock, self).__init__()
        self.filters = filters
        self.conv1 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=2)
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')

        self.conv2 = nn.Conv2D(self.filters, kernel_size=3, padding=1, strides=1)
        self.bn2 = nn.BatchNorm()
        self.act2 = nn.Activation('relu')

        self.conv3 = nn.Conv2D(self.filters * 2, kernel_size=3 ,padding =1,strides=1)

        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)

    def hybrid_forward(self, F, x):
        stem1 = self.act1(self.bn1(self.conv1(x)))
        stem2 = self.act2(self.bn2(self.conv2(stem1)))
        stem3 = self.conv3(stem2)
        out = self.pool(stem3)
        return out

class conv_block(nn.HybridBlock):
    def __init__(self, filters):
        super(conv_block, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, padding=1)
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class DenseBlcok(nn.HybridBlock):
    def __init__(self, num_convs, num_channels):  # layers, growth rate
        super(DenseBlcok, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            for _ in range(num_convs):
                self.net.add(
                    conv_block(num_channels)
                )

    def hybrid_forward(self, F, x):
        for blk in self.net:
            Y = blk(x)
            x = F.concat(x, Y, dim=1)

        return x

class transitionLayer(nn.HybridBlock):
    def __init__(self, filters, with_pool=True):
        super(transitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm()
        self.act1 = nn.Activation('relu')
        self.conv1 = nn.Conv2D(filters, kernel_size=1)
        self.with_pool = with_pool
        if self.with_pool:
            self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)

    def hybrid_forward(self, F, x):
        out = self.conv1(self.act1(self.bn1(x)))
        if self.with_pool:
            out = self.pool(out)
        return out

class conv_conv(nn.HybridBlock):
    def __init__(self, filters):
        super(conv_conv, self).__init__()
        self.net = nn.HybridSequential()
        with self.net.name_scope():
            self.net.add(
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=1, strides=1),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(filters, kernel_size=3, strides=2, padding=1)
            )

    def hybrid_forward(self, F, x):
        return self.net(x)

class pool_conv(nn.HybridBlock):
    def __init__(self, filters):
        super(pool_conv, self).__init__()
        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=2)
        self.bn = nn.BatchNorm()
        self.act = nn.Activation('relu')
        self.conv = nn.Conv2D(filters, kernel_size=1)

    def hybrid_forward(self, F, x):
        out = self.conv(self.act(self.bn(self.pool(x))))
        return out

class cls_predictor(nn.HybridBlock):  # (num_anchors * (num_classes + 1), 3, padding=1)
    def __init__(self, num_anchors, num_classes):
        super(cls_predictor, self).__init__()
        self.class_predcitor = nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

    def hybrid_forward(self, F, x):
        return self.class_predcitor(x)

class bbox_predictor(nn.HybridBlock):  # (num_anchors * 4, 3, padding=1)
    def __init__(self, num_anchors):
        super(bbox_predictor, self).__init__()
        self.bbox_predictor = nn.Conv2D(num_anchors * 4, 3, padding=1)

    def hybrid_forward(self, F, x):
        return self.bbox_predictor(x)

class DSOD(nn.HybridBlock):
    def __init__(self, stem_filter, num_init_layer, growth_rate, factor, num_class):
        super(DSOD, self).__init__()
        if factor == 0.5:
            self.factor = 2
        else:
            self.factor = 1
        self.num_cls = num_class
        self.sizes = [[.2, .2], [.37, .37], [.45, .45], [.54, .54], [.71, .71], [.88, .88]]  #
        self.ratios = [[1, 2, 0.5]] * 6
        self.num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        trans1_filter = ((stem_filter * 2) + (num_init_layer * growth_rate) // self.factor)

        self.backbone_fisrthalf = nn.HybridSequential()
        with self.backbone_fisrthalf.name_scope():
            self.backbone_fisrthalf.add(
                stemblock(stem_filter),
                DenseBlcok(6, growth_rate),
                transitionLayer(trans1_filter),
                DenseBlcok(8, growth_rate)

            )
        trans2_filter = ((trans1_filter) + (8 * growth_rate) // self.factor)
        trans3_filter = ((trans2_filter) + (8 * growth_rate) // self.factor)

        self.backbone_secondehalf = nn.HybridSequential()
        with self.backbone_secondehalf.name_scope():
            self.backbone_secondehalf.add(
                transitionLayer(trans2_filter),
                DenseBlcok(8, growth_rate),
                transitionLayer(trans3_filter, with_pool=False),
                DenseBlcok(8, growth_rate),
                transitionLayer(256, with_pool=False)
            )
        self.PC_layer = nn.HybridSequential()  # pool -> conv
        numPC_layer = [256, 256, 128, 128, 128]
        with self.PC_layer.name_scope():
            for i in range(5):
                self.PC_layer.add(
                    pool_conv(numPC_layer[i]),
                )
        self.CC_layer = nn.HybridSequential()  # conv1 -> conv3
        numCC_layer = [256, 128, 128, 128]
        with self.CC_layer.name_scope():
            for i in range(4):
                self.CC_layer.add(
                    conv_conv(numCC_layer[i])
                )

        self.class_predictors = nn.HybridSequential()
        with self.class_predictors.name_scope():
            for _ in range(6):
                self.class_predictors.add(
                    cls_predictor(self.num_anchors, self.num_cls)
                )

        self.box_predictors = nn.HybridSequential()
        with self.box_predictors.name_scope():
            for _ in range(6):
                self.box_predictors.add(
                    bbox_predictor(self.num_anchors)
                )

    def flatten_prediction(self, pred):
        return pred.transpose(axes=(0, 2, 3, 1)).flatten()

    def concat_predictions(self, preds):
        return nd.concat(*preds, dim=1)

    def hybrid_forward(self, F, x):

        anchors, class_preds, box_preds = [], [], []

        scale_1 = self.backbone_fisrthalf(x)

        anchors.append(MultiBoxPrior(
            scale_1, sizes=self.sizes[0], ratios=self.ratios[0]))
        class_preds.append(
            self.flatten_prediction(self.class_predictors[0](scale_1)))
        box_preds.append(
            self.flatten_prediction(self.box_predictors[0](scale_1)))

        out = self.backbone_secondehalf(scale_1)
        PC_1 = self.PC_layer[0](scale_1)
        scale_2 = F.concat(out, PC_1, dim=1)

        anchors.append(MultiBoxPrior(
            scale_2, sizes=self.sizes[1], ratios=self.ratios[1]))
        class_preds.append(
            self.flatten_prediction(self.class_predictors[1](scale_2)))
        box_preds.append(
            self.flatten_prediction(self.box_predictors[1](scale_2)))

        scale_predict = scale_2
        for i in range(1, 5):
            PC_Predict = self.PC_layer[i](scale_predict)
            CC_Predict = self.CC_layer[i - 1](scale_predict)
            scale_predict = F.concat(PC_Predict, CC_Predict, dim=1)

            anchors.append(MultiBoxPrior(
                scale_predict, sizes=self.sizes[i + 1], ratios=self.ratios[i + 1]))
            class_preds.append(
                self.flatten_prediction(self.class_predictors[i + 1](scale_predict)))
            box_preds.append(
                self.flatten_prediction(self.box_predictors[i + 1](scale_predict)))

        # print(scale_predict.shape)

        anchors = self.concat_predictions(anchors)
        class_preds = self.concat_predictions(class_preds)
        box_preds = self.concat_predictions(box_preds)

        class_preds = class_preds.reshape(shape=(0, -1, self.num_cls + 1))

        return anchors, class_preds, box_preds
