#author: shehab el deen ayman mounir
import tensorflow as tensorflow
import numpy as numpy
import yolo.networkConfigiration as config

class yolo_v2(object):
    def __init__(self, isTraining = True):
        self.classes = config.CLASSES
        self.num_class = len(self.classes)
        self.BoxesPerCell = config.BOX_PRE_CELL
        self.CellSize = config.CELL_SIZE
        self.BatchSize = config.BATCH_SIZE
        self.ImageSize = config.IMAGE_SIZE
        self.Anchor = config.ANCHOR
        self.Alpha = config.ALPHA

        self.ScaleOfClass = 1.0
        self.ScaleOfObject = 5.0
        self.NoOfObjectScale = 1.0
        self.ScaleCoordinate = 1.0
        
        self.offset = numpy.transpose(numpy.reshape(numpy.array([numpy.arange(self.CellSize)] * self.CellSize * self.BoxesPerCell),
                                         [self.BoxesPerCell, self.CellSize, self.CellSize]), (1, 2, 0))
        self.offset = tensorflow.reshape(tensorflow.constant(self.offset, dtype=tensorflow.float32), [1, self.CellSize, self.CellSize, self.BoxesPerCell])
        self.offset = tensorflow.tile(self.offset, (self.BatchSize, 1, 1, 1))

        self.images = tensorflow.placeholder(tensorflow.float32, [None, self.ImageSize, self.ImageSize, 3], name='images')
        self.logits = self.NetworkStructure(self.images)

        if isTraining:
            self.labels = tensorflow.placeholder(tensorflow.float32, [None, self.CellSize, self.CellSize, self.BoxesPerCell, self.num_class + 5], name = 'labels')
            self.total_loss = self.LossLayer(self.logits, self.labels)
            tensorflow.summary.scalar('total_loss', self.total_loss)

    def NetworkStructure(self, inputs):
        Network = self.ConvolutionLayer(inputs, [3, 3, 3, 32], name = 'ConvolutionLayer1')
        Network = self.PoolingLayer(Network, name = 'PoolingLayer1')
        Network = self.ConvolutionLayer(Network, [3, 3, 32, 64], name = 'ConvolutionLayer2')
        Network = self.PoolingLayer(Network, name = 'PoolingLayer2')
        Network = self.ConvolutionLayer(Network, [3, 3, 64, 128], name = 'ConvolutionLayer3')
        Network = self.ConvolutionLayer(Network, [1, 1, 128, 64], name = 'ConvolutionLayer4')
        Network = self.ConvolutionLayer(Network, [3, 3, 64, 128], name = 'ConvolutionLayer5')
        Network = self.PoolingLayer(Network, name = 'PoolingLayer3')
        Network = self.ConvolutionLayer(Network, [3, 3, 128, 256], name = 'ConvolutionLayer6')
        Network = self.ConvolutionLayer(Network, [1, 1, 256, 128], name = 'ConvolutionLayer7')
        Network = self.ConvolutionLayer(Network, [3, 3, 128, 256], name = 'ConvolutionLayer8')
        Network = self.PoolingLayer(Network, name = 'PoolingLayer4')
        Network = self.ConvolutionLayer(Network, [3, 3, 256, 512], name = 'ConvolutionLayer9')
        Network = self.ConvolutionLayer(Network, [1, 1, 512, 256], name = 'ConvolutionLayer10')
        Network = self.ConvolutionLayer(Network, [3, 3, 256, 512], name = 'ConvolutionLayer11')
        Network = self.ConvolutionLayer(Network, [1, 1, 512, 256], name = 'ConvolutionLayer12')
        net16 = self.ConvolutionLayer(Network, [3, 3, 256, 512], name = 'ConvolutionLayer13')
        Network = self.PoolingLayer(net16, name = 'PoolingLayer5')
        Network = self.ConvolutionLayer(Network, [3, 3, 512, 1024], name = 'ConvolutionLayer14')
        Network = self.ConvolutionLayer(Network, [1, 1, 1024, 512], name = 'ConvolutionLayer15')
        Network = self.ConvolutionLayer(Network, [3, 3, 512, 1024], name = 'ConvolutionLayer16')
        Network = self.ConvolutionLayer(Network, [1, 1, 1024, 512], name = 'ConvolutionLayer17')
        Network = self.ConvolutionLayer(Network, [3, 3, 512, 1024], name = 'ConvolutionLayer18')
        Network = self.ConvolutionLayer(Network, [3, 3, 1024, 1024], name = 'ConvolutionLayer19')
        net24 = self.ConvolutionLayer(Network, [3, 3, 1024, 1024], name = 'ConvolutionLayer20')
        Network = self.ConvolutionLayer(net16, [1, 1, 512, 64], name = 'ConvolutionLayer21')
        Network = self.ReorganizeNetwork(Network)
        Network = tensorflow.concat([Network, net24], 3)
        Network = self.ConvolutionLayer(Network, [3, 3, int(Network.get_shape()[3]), 1024], name = 'ConvolutionLayer22')
        Network = self.ConvolutionLayer(Network, [1, 1, 1024, self.BoxesPerCell * (self.num_class + 5)], batch_norm=False, name = 'ConvolutionLayer23')

        return Network


    def ConvolutionLayer(self, inputs, shape, batch_norm = True, name = '0_conv'):
        weight = tensorflow.Variable(tensorflow.truncated_normal(shape, stddev=0.1), name='weight')
        biases = tensorflow.Variable(tensorflow.constant(0.1, shape=[shape[3]]), name='biases')

        Convolution = tensorflow.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)

        if batch_norm:
            depth = shape[3]
            scale = tensorflow.Variable(tensorflow.ones([depth, ], dtype='float32'), name='scale')
            shift = tensorflow.Variable(tensorflow.zeros([depth, ], dtype='float32'), name='shift')
            mean = tensorflow.Variable(tensorflow.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tensorflow.Variable(tensorflow.ones([depth, ], dtype='float32'), name='rolling_variance')
            conv_bn = tensorflow.nn.batch_normalization(Convolution, mean, variance, shift, scale, 1e-05)
            Convolution = tensorflow.add(conv_bn, biases)
            Convolution = tensorflow.maximum(self.Alpha * Convolution, Convolution)
        else:
            Convolution = tensorflow.add(Convolution, biases)

        return Convolution


    def PoolingLayer(self, inputs, name = '1_pool'):
        pool = tensorflow.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
        return pool


    def ReorganizeNetwork(self, inputs):
        outputs_1 = inputs[:, ::2, ::2, :]
        outputs_2 = inputs[:, ::2, 1::2, :]
        outputs_3 = inputs[:, 1::2, ::2, :]
        outputs_4 = inputs[:, 1::2, 1::2, :]
        output = tensorflow.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis = 3)
        return output


    def LossLayer(self, predict, label):
        predict = tensorflow.reshape(predict, [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell, self.num_class + 5])
        CoordinateBox = tensorflow.reshape(predict[:, :, :, :, :4], [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell, 4])
        ConfidenceBox = tensorflow.reshape(predict[:, :, :, :, 4], [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell, 1])
        ClassBox = tensorflow.reshape(predict[:, :, :, :, 5:], [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell, self.num_class])
        boxes1 = tensorflow.stack([(1.0 / (1.0 + tensorflow.exp(-1.0 * CoordinateBox[:, :, :, :, 0])) + self.offset) / self.CellSize, (1.0 / (1.0 + tensorflow.exp(-1.0 * CoordinateBox[:, :, :, :, 1])) + tensorflow.transpose(self.offset, (0, 2, 1, 3))) / self.CellSize,  tensorflow.sqrt(tensorflow.exp(CoordinateBox[:, :, :, :, 2]) * numpy.reshape(self.Anchor[:5], [1, 1, 1, 5]) / self.CellSize),tensorflow.sqrt(tensorflow.exp(CoordinateBox[:, :, :, :, 3]) * numpy.reshape(self.Anchor[5:], [1, 1, 1, 5]) / self.CellSize)])
        CoordinateBoxTranspose = tensorflow.transpose(boxes1, (1, 2, 3, 4, 0))
        ConfidenceBox = 1.0 / (1.0 + tensorflow.exp(-1.0 * ConfidenceBox))
        ClassBox = tensorflow.nn.softmax(ClassBox)
        Response = tensorflow.reshape(label[:, :, :, :, 0], [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell])
        Boxes = tensorflow.reshape(label[:, :, :, :, 1:5], [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell, 4])
        classes = tensorflow.reshape(label[:, :, :, :, 5:], [self.BatchSize, self.CellSize, self.CellSize, self.BoxesPerCell, self.num_class])
        IntersectionOverUnion = self.CalculateIntersectionOverUnion(CoordinateBoxTranspose, Boxes)
        BestBoundingBox = tensorflow.to_float(tensorflow.equal(IntersectionOverUnion, tensorflow.reduce_max(IntersectionOverUnion, axis=-1, keep_dims=True)))
        confs = tensorflow.expand_dims(BestBoundingBox * Response, axis = 4)
        conid = self.NoOfObjectScale * (1.0 - confs) + self.ScaleOfObject * confs
        cooid = self.ScaleCoordinate * confs
        proid = self.ScaleOfClass * confs
        coo_loss = cooid * tensorflow.square(CoordinateBoxTranspose - Boxes)
        con_loss = conid * tensorflow.square(ConfidenceBox - confs)
        pro_loss = proid * tensorflow.square(ClassBox - classes)
        loss = tensorflow.concat([coo_loss, con_loss, pro_loss], axis = 4)
        loss = tensorflow.reduce_mean(tensorflow.reduce_sum(loss, axis = [1, 2, 3, 4]), name = 'loss')
        return loss


    def CalculateIntersectionOverUnion(self, boxes1, boxes2):
        boxx = tensorflow.square(boxes1[:, :, :, :, 2:4])
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tensorflow.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes1 = tensorflow.transpose(box, (1, 2, 3, 4, 0))
        boxx = tensorflow.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tensorflow.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5, boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tensorflow.transpose(box, (1, 2, 3, 4, 0))
        left_up = tensorflow.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tensorflow.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
        intersection = tensorflow.maximum(right_down - left_up, 0.0)
        SquareOfIntersection = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        SquareOfUnion = boxes1_square + boxes2_square - SquareOfIntersection
        return tensorflow.clip_by_value(1.0 * SquareOfIntersection / SquareOfUnion, 0.0, 1.0)
