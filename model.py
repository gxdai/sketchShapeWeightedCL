import tensorflow as tf
import numpy as np
import numpy.matlib
from scipy.spatial import distance
import os
import sys
from data import Dataset
import time
from RetrievalEvaluation import RetrievalEvaluation
#tf.set_random_seed(222)
#np.random.seed(222)

class model(Dataset):
    """ Create the model for weighted contrastive loss"""
    def __init__(self, lamb=20., ckpt_dir='./checkpoint', ckpt_name='model',
            batch_size=30, margin=10., learning_rate=0.0001, momentum=0.9, sketch_train_list=None,
            sketch_test_list=None, shape_list=None, num_views_shape=12, class_num=90, normFlag=0,
            logdir=None, lossType='contrastiveLoss', activationType='relu', phase='train', inputFeaSize=4096, outputFeaSize=100, maxiter=100000):

        """
        lamb:           The parameter for sinkhorn iteration
        ckpt_dir:       The directory for saving checkpoint file
        ckpt_name:      The name of checkpoint file
        batch_size:     The training batch_size
        margin:         The margin for contrastive loss
        learning_rate:  The learning rate
        momentum:       The momentum
        sketch_train_list:  The list file of training sketch
        sketch_test_list:   The list file of testing sketch
        shape_list:         The list file of shape
        num_views_shape:    The total number of shape views
        class_num:          The total nubmer of classes
        normFlag:           1 for normalizing input features, 0 for not normalizing input features
        logdir:             The log directory
        lossType:           choosing the loss function
        activationType:     The activation function
        phase:              choosing training or testing
        inputFeaSize:       The dimension of input features
        outputFeaSize:      The dimension of output features
        maxiter:            The maximum number of iterations
        """

        self.lamb               =       lamb
        self.ckpt_dir           =       ckpt_dir
        self.ckpt_name          =       ckpt_name
        self.batch_size         =       batch_size
        self.logdir             =       logdir
        self.class_num          =       class_num
        self.num_views_shape    =       num_views_shape
        self.maxiter            =       maxiter
        self.inputFeaSize       =       inputFeaSize
        self.outputFeaSize      =       outputFeaSize
        self.margin             =       margin
        self.learning_rate      =       learning_rate
        self.momentum           =       momentum
        self.lossType           =       lossType
        self.phase              =       phase
        self.normFlag           =       normFlag
        self.activationType     =       activationType

        print("self.lamb               =       {:2.5f}".format(self.lamb))
        print("self.ckpt_dir           =       {:10}".format(self.ckpt_dir))
        print("self.ckpt_name          =       {:10}".format(self.ckpt_name))
        print("self.batch_size         =       {:5d}".format(self.batch_size))
        print("self.logdir             =       {:10}".format(self.logdir))
        print("self.class_num          =       {:5d}".format(self.class_num))
        print("self.num_views_shape    =       {:5d}".format(self.num_views_shape))
        print("self.maxiter            =       {:5d}".format(self.maxiter))
        print("self.inputFeaSize       =       {:5d}".format(self.inputFeaSize))
        print("self.outputFeaSize      =       {:5d}".format(self.outputFeaSize))
        print("self.margin             =       {:2.5f}".format(self.margin))
        print("self.learning_rate      =       {:2.5f}".format(self.learning_rate))
        print("self.momentum           =       {:2.5f}".format(self.momentum))
        print("self.lossType           =       {:10}".format(self.lossType))
        print("self.phase              =       {:10}".format(self.phase))
        print("self.normFlag           =       {:10d}".format(self.normFlag))
        print("self.activationType     =       {:10}".format(self.activationType))


        # class inheritance from Dataset class
        Dataset.__init__(self,sketch_train_list=sketch_train_list, sketch_test_list=sketch_test_list, shape_list=shape_list, num_views_shape=num_views_shape, feaSize=inputFeaSize, class_num=class_num, phase=phase, normFlag=normFlag)
        print("INIT")

        self.build_model()
        print("debug#################################")

    def sketchNetwork(self, x):          #### for sketch network
        if self.activationType == 'relu':
            stddev = 0.01
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.relu(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 500, "fc3", 0.1)
            fc4 = self.fc_layer(fc3, self.outputFeaSize, "fc4", 0.1)

            return fc4, fc3
        elif self.activationType == 'sigmoid':
            stddev = 0.1
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.sigmoid(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.sigmoid(fc2)
            fc3 = self.fc_layer(ac2, self.outputFeaSize, "fc3", stddev)
            ac3 = tf.nn.sigmoid(fc3)

            return ac3, ac2

    def shapeNetwork(self, x):          #### for sketch network
        if self.activationType == 'relu':
            stddev = 0.01
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.relu(fc1)
            fc2 = self.fc_layer(ac1, 2000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 1000, "fc3", stddev)
            ac3 = tf.nn.relu(fc3)
            fc4 = self.fc_layer(ac3, 500, "fc4", 0.1)
            fc5 = self.fc_layer(fc4, self.outputFeaSize, "fc5", 0.1)

            return fc5, fc4
        elif self.activationType == 'sigmoid':
            stddev = 0.1
            fc1 = self.fc_layer(x, 2000, "fc1", stddev)
            ac1 = tf.nn.sigmoid(fc1)
            fc2 = self.fc_layer(ac1, 1000, "fc2", stddev)
            ac2 = tf.nn.relu(fc2)
            fc3 = self.fc_layer(ac2, 500, "fc3", stddev)
            ac3 = tf.nn.sigmoid(fc3)
            fc4 = self.fc_layer(ac3, self.outputFeaSize, "fc4", stddev)
            ac4 = tf.nn.sigmoid(fc4)

            return ac4, ac3

    def fc_layer(self, bottom, n_weight, name, stddev):
        #assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[-1]
        #initer = tf.truncated_normal_initializer(stddev=0.01)
        initer = tf.truncated_normal_initializer(stddev=stddev)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        ## rescale weight by 0.1
        #    W = tf.mul(.1, W)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.0, shape=[n_weight], dtype=tf.float32))
        ## rescale biase by 0.1
        #    b = tf.mul(.1, b)
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc

    def calculateGroundMetric(self, sketch_fea, shape_fea):
        """
        calculate the ground metric between sketch and shape
        """
        square_sketch_fea = tf.reduce_sum(tf.square(sketch_fea), axis=2)
        square_sketch_fea = tf.expand_dims(square_sketch_fea, axis=2)

        square_shape_fea = tf.reduce_sum(tf.square(shape_fea), axis=2)

        square_shape_fea = tf.expand_dims(square_shape_fea, axis=1)

        correlationTerm = tf.matmul(sketch_fea, tf.transpose(shape_fea, perm=[0, 2, 1]))

        groundMetricRaw = tf.add(tf.subtract(square_sketch_fea, tf.multiply(2., correlationTerm)), square_shape_fea)

        # Flatten the groundMetric as a vector
        groundMetricRaw = tf.reshape(groundMetricRaw, [self.batch_size, -1])

        return groundMetricRaw

    def calculateGroundMetricContrastive(self, batchFea1, batchFea2, labelMatrix):
        """
        calculate the ground metric between sketch and shape
        """
        print("Calculate the ground metrics between two batches of features")
        print("batchFea1.get_shape().as_list() = {}".format(batchFea1.get_shape().as_list()))
        print("batchFea2.get_shape().as_list() = {}".format(batchFea2.get_shape().as_list()))
        print("labelMatrix.get_shape().as_list() = {}".format(labelMatrix.get_shape().as_list()))
        squareBatchFea1 = tf.reduce_sum(tf.square(batchFea1), axis=1)
        squareBatchFea1 = tf.expand_dims(squareBatchFea1, axis=1)

        squareBatchFea2 = tf.reduce_sum(tf.square(batchFea2), axis=1)
        squareBatchFea2 = tf.expand_dims(squareBatchFea2, axis=0)

        correlationTerm = tf.matmul(batchFea1, tf.transpose(batchFea2, perm=[1, 0]))

        groundMetric = tf.add(tf.subtract(squareBatchFea1, tf.multiply(2., correlationTerm)), squareBatchFea2)


        # Get ground metric cost for negative pair
        hinge_groundMetric = tf.maximum(0., self.margin -  groundMetric)


        GM_positivePair = tf.multiply(labelMatrix, groundMetric)
        GM_negativePair = tf.multiply(1 - labelMatrix, hinge_groundMetric)
        GM = tf.add(GM_positivePair, GM_negativePair)

        expGM = tf.exp(tf.multiply(-1., GM))    # This is for optimizing "T"

        # Flatten the groundMetric as a vector
        GMFlatten = tf.reshape(GM, [-1])
        print("expGM.get_shape().as_list() = {}".format(expGM.get_shape().as_list()))

        return GMFlatten, expGM



    def groundMetricBatch(self, sketch_fea, shape_fea):
        """
        calculate the ground metric between sketch and shape
        """
        print(self.batch_size)

        square_sketch_fea = tf.reduce_sum(tf.square(sketch_fea), axis=1)
        square_sketch_fea = tf.expand_dims(square_sketch_fea, axis=1)

        square_shape_fea = tf.reduce_sum(tf.square(shape_fea), axis=1)

        square_shape_fea = tf.expand_dims(square_shape_fea, axis=0)

        correlationTerm = tf.matmul(sketch_fea, tf.transpose(shape_fea, perm=[1, 0]))

        groundMetric = tf.add(tf.subtract(square_sketch_fea, tf.multiply(2., correlationTerm)), square_shape_fea)

        return groundMetric



    def simLabelGeneration(self, label1, label2):
        label1 = tf.tile(label1, [1, self.batch_size])
        label2 = tf.tile(label2, [self.batch_size, 1])
        simLabel = tf.cast(tf.equal(tf.reshape(label1, [-1]), tf.reshape(label2, [-1])), tf.float32)
        simLabelMatrix = tf.reshape(simLabel, [self.batch_size, self.batch_size])

        return simLabelMatrix



    def groundMetricBatchWithLabel(self, sketch_fea, shape_fea, sketch_label, shape_label):
        """
        calculate the ground metric between sketch and shape
        """
        print(self.batch_size)

        simLabelMatrix = self.simLabelGeneration(sketch_label, shape_label)

        square_sketch_fea = tf.reduce_sum(tf.square(sketch_fea), axis=1)
        square_sketch_fea = tf.expand_dims(square_sketch_fea, axis=1)

        square_shape_fea = tf.reduce_sum(tf.square(shape_fea), axis=1)

        square_shape_fea = tf.expand_dims(square_shape_fea, axis=0)

        correlationTerm = tf.matmul(sketch_fea, tf.transpose(shape_fea, perm=[1, 0]))


        groundMetric = tf.add(tf.subtract(square_sketch_fea, tf.multiply(2., correlationTerm)), square_shape_fea)

        hinge_groundMetric = tf.maximum(0., self.margin -  groundMetric)


        GM_positivePair = tf.multiply(simLabelMatrix, groundMetric)
        GM_negativePair = tf.multiply(1 - simLabelMatrix, hinge_groundMetric)

        GM = tf.add(GM_positivePair, GM_negativePair)

        expGM = tf.exp(tf.multiply(-1., GM))    # This is for optimizing "T"

        # Flatten the groundMetric as a vector
        GMFlatten = tf.reshape(GM, [-1])


        return GM



    def sinkhornIter(self, groundMetric, name='sinkhorn'):
        with tf.variable_scope(name) as scope:
            #u0 = tf.constant(1./self.gmWidth, shape=[self.gmWidth, 1], name='u0')
            u0 = tf.constant(1./self.batch_size, shape=[self.batch_size, 1], name='u0')
            print("groundMetric.get_shape().as_list() = {}".format(groundMetric.get_shape().as_list()))
            groundMetric_ = tf.transpose(groundMetric, perm=[1, 0])      # transpose the ground metric

            epsilon=1e-12

            u = tf.get_variable(name='u', shape=[self.batch_size, 1], initializer=tf.constant_initializer(1.0))
            v = tf.get_variable(name='v', shape=[self.batch_size, 1], initializer=tf.constant_initializer(1.0))

            v_assign_op = v.assign(tf.div(u0, tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric_)), u)+epsilon))
            u_assign_op = u.assign(tf.div(u0, tf.matmul(tf.exp(tf.multiply(-self.lamb, groundMetric)), v)+epsilon))
            u_reset = u.assign(tf.constant(1.0, shape=[self.batch_size, 1], name='reset_u'))


            T = tf.matmul(tf.matmul(tf.diag(tf.reshape(u, [-1])), tf.exp(tf.multiply(-self.lamb, groundMetric))), tf.diag(tf.reshape(v, [-1])), name='T')
            T_flatten = tf.reshape(T, [-1])


            debug = tf.reduce_sum(T)

            return v_assign_op, u_assign_op, u, v, T, T_flatten, u_reset, debug



    def contrastiveLoss(self, input_sketch, input_shape):

        pairwiseDistance = tf.reduce_sum(tf.square(input_sketch - input_shape), axis=2, name='pairwiseDistance')
        hinge_distance = tf.maximum(0., tf.subtract(self.margin, pairwiseDistance))


        print(pairwiseDistance.get_shape())
        print(hinge_distance.get_shape())
        print(self.simLabel.get_shape())
        self.positivePairDistance = tf.multiply(self.simLabel, pairwiseDistance)
        self.negativePairDistance = tf.multiply(1 - self.simLabel, hinge_distance)

        self.contrastiveDistance = self.positivePairDistance + self.negativePairDistance

        self.loss = tf.reduce_mean(tf.reduce_sum(self.contrastiveDistance, axis=1), axis=0)
        self.loss_summary = tf.summary.scalar('loss', self.loss)






    def doubleContrastiveLoss(self):

        def contrastiveLoss(input_fea_1, input_fea_2, simLabel, margin, lossName):
            # contrastive loss construction
            distance_positive = tf.reduce_sum(tf.square(input_fea_1 - input_fea_2), axis=2)
            distance_negative = tf.maximum(0., margin-distance_positive)
            distance_contrastive = tf.add(tf.multiply(simLabel, distance_positive), tf.multiply(1. - simLabel, distance_negative))

            loss = tf.reduce_mean(tf.reduce_mean(distance_contrastive, axis=1), axis=0, name=lossName)
            loss_summary = tf.summary.scalar(lossName, loss)

            return loss, loss_summary, distance_contrastive, distance_positive



        # contrastive loss for sketch
        print(self.sketch_1.get_shape())
        print(self.sketch_2.get_shape())
        print(self.simLabel_sketch.get_shape())
        self.loss_sketch, self.loss_sketch_summary, _, _ = contrastiveLoss(self.sketch_1, self.sketch_2, self.simLabel_sketch, self.margin, 'loss_sketch')

        # contrastive loss for shape
        self.loss_shape, self.loss_shape_summary, _, _ = contrastiveLoss(self.shape_1, self.shape_2, self.simLabel_shape, self.margin, 'loss_shape')


        # contrastive loss for sketch-shape 1
        self.loss_cross_1, self.loss_cross_summary_1, self.distance_cross_1, self.distance_positive_1 = contrastiveLoss(self.sketch_1, self.shape_1, self.simLabel_cross_1, self.margin, 'loss_cross_1')

        # contrastive loss for sketch-shape 2
        self.loss_cross_2, self.loss_cross_summary_2, self.distance_cross_2, self.distance_positive_2 = contrastiveLoss(self.sketch_2, self.shape_2, self.simLabel_cross_2, self.margin, 'loss_cross_2')

        # summation of all the loss

        # The summary loss should be seprated for different loss types




    def wassersteinLoss(self, input_sketch, input_shape):

        self.groundMetricFlatten, self.groundMetric = self.calculateGroundMetricContrastive(input_sketch, input_shape)

        # sinkhorn iter
        self.v_assign_op, self.u_assign_op, self.u, self.v, self.T, self.T_flatten, self.u_reset, self.debug = self.sinkhornIter(self.groundMetric)

        # variable list for ground metric network

        self.loss = tf.reduce_sum(tf.multiply(self.groundMetricFlatten, self.T_flatten), name='loss')
        self.loss_summary = tf.summary.scalar('loss', self.loss)


    def build_network(self):
        with tf.variable_scope('sketch') as scope:
            self.output_sketch_fea_1, self.output_sketch_debug_1 = self.sketchNetwork(self.input_sketch_fea_1)
            scope.reuse_variables()
            self.output_sketch_fea_2, self.output_sketch_debug_2 = self.sketchNetwork(self.input_sketch_fea_2)

        with tf.variable_scope('shape') as scope:
            self.output_shape_fea_1, self.output_shape_debug_1 = self.shapeNetwork(self.input_shape_fea_1)
            scope.reuse_variables()
            self.output_shape_fea_2, self.output_shape_debug_2 = self.shapeNetwork(self.input_shape_fea_2)


    def build_model(self):
        # input sketch placeholder

        self.input_sketch_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize], name='input_sketch_fea_1')
        self.input_sketch_label_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_sketch_label_1')
        self.input_sketch_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize], name='input_sketch_fea_2')
        self.input_sketch_label_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_sketch_label_2')

        # input shape placeholder
        self.input_shape_fea_1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize], name='input_shape_fea_1')
        self.input_shape_label_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_shape_label_1')
        self.input_shape_fea_2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputFeaSize], name='input_shape_fea_2')
        self.input_shape_label_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name='input_shape_label_2')


        # similarity matrix for sketch
        self.simLabel_sketch = self.simLabelGeneration(self.input_sketch_label_1, self.input_sketch_label_2)

        # similarity matrix for shape
        self.simLabel_shape = self.simLabelGeneration(self.input_shape_label_1, self.input_shape_label_2)



        # similarity matrix for cross 1
        self.simLabel_cross_1 = self.simLabelGeneration(self.input_sketch_label_1, self.input_shape_label_1)

        # similarity matrix for cross 2
        self.simLabel_cross_2 = self.simLabelGeneration(self.input_sketch_label_2, self.input_shape_label_2)


        self.build_network()


        print(self.lossType)

        if self.lossType == 'doubleContrastiveLoss':
            # just for visualization purpose
            self.groundMetric_1 = self.groundMetricBatch(self.sketch_out_1, self.shape_out_1)
            self.groundMetric_2 = self.groundMetricBatch(self.sketch_out_2, self.shape_out_2)
            ################################################################################

            # use contrastive loss within each domain and across different domains
            self.sketch_1 = tf.reshape(self.sketch_out_1, [-1, self.outputFeaSize])
            self.sketch_2 = tf.reshape(self.sketch_out_2, [-1, self.outputFeaSize])


            self.shape_1 = tf.reshape(self.shape_out_1, [-1, self.outputFeaSize])
            self.shape_2 = tf.reshape(self.shape_out_2, [-1, self.outputFeaSize])

            self.doubleContrastiveLoss()



            self.loss = tf.add_n([self.loss_sketch, self.loss_shape, self.loss_cross_1, self.loss_cross_2], name='loss')
            self.loss_summary = tf.summary.scalar('loss', self.loss)


        elif self.lossType == 'weightedContrastiveLoss':

            print("Caculating the ground matrix")
            self.GM_sketch, self.expGM_sketch = self.calculateGroundMetricContrastive(self.output_sketch_fea_1, self.output_sketch_fea_2, self.simLabel_sketch)
            self.GM_shape, self.expGM_shape = self.calculateGroundMetricContrastive(self.output_shape_fea_1, self.output_shape_fea_2, self.simLabel_shape)
            self.GM_cross_1, self.expGM_cross_1 = self.calculateGroundMetricContrastive(self.output_sketch_fea_1, self.output_shape_fea_1, self.simLabel_cross_1)
            self.GM_cross_2, self.expGM_cross_2 = self.calculateGroundMetricContrastive(self.output_sketch_fea_2, self.output_shape_fea_2, self.simLabel_cross_2)



            print("Sinkhorn iteration construction")

            self.sketch_v_assign_op, self.sketch_u_assign_op, self.sketch_u, self.sketch_v, self.sketch_T, self.sketch_T_flatten, self.sketch_u_reset, self.sketch_debug = self.sinkhornIter(self.expGM_sketch, name='sinkhorn_sketch')
            self.shape_v_assign_op, self.shape_u_assign_op, self.shape_u, self.shape_v, self.shape_T, self.shape_T_flatten, self.shape_u_reset, self.shape_debug = self.sinkhornIter(self.expGM_shape, name='sinkhorn_shape')
            self.cross_1_v_assign_op, self.cross_1_u_assign_op, self.cross_1_u, self.cross_1_v, self.cross_1_T, self.cross_1_T_flatten, self.cross_1_u_reset, self.cross_1_debug = self.sinkhornIter(self.expGM_cross_1, name='sinkhorn_cross_1')
            self.cross_2_v_assign_op, self.cross_2_u_assign_op, self.cross_2_u, self.cross_2_v, self.cross_2_T, self.cross_2_T_flatten, self.cross_2_u_reset, self.cross_2_debug = self.sinkhornIter(self.expGM_cross_2, name='sinkhorn_cross_2')



            # create loss

            self.loss_sketch = tf.reduce_sum(tf.multiply(self.GM_sketch, self.sketch_T_flatten), name='sketch_loss')
            self.loss_shape = tf.reduce_sum(tf.multiply(self.GM_shape, self.shape_T_flatten), name='shape_loss')
            self.loss_cross_1 = tf.reduce_sum(tf.multiply(self.GM_cross_1, self.cross_1_T_flatten), name='cross_1_loss')
            self.loss_cross_2 = tf.reduce_sum(tf.multiply(self.GM_cross_2, self.cross_2_T_flatten), name='cross_2_loss')
            self.loss = tf.add_n([self.loss_sketch, self.loss_shape, self.loss_cross_1, self.loss_cross_2], name='loss')
            # distance used for retrieval


            self.loss_summary_sketch = tf.summary.scalar('sketch_loss', self.loss_sketch)
            self.loss_summary_shape = tf.summary.scalar('shape_loss', self.loss_shape)
            self.loss_summary_cross_1 = tf.summary.scalar('cross_loss_1', self.loss_cross_1)
            self.loss_summary_cross_2 = tf.summary.scalar('cross_loss_2', self.loss_cross_2)
            self.loss_summary = tf.summary.scalar('loss', self.loss)


        elif self.lossType == 'contrastiveLoss':

            print("Caculating the ground matrix")
            self.GM_sketch, self.expGM_sketch = self.calculateGroundMetricContrastive(self.output_sketch_fea_1, self.output_sketch_fea_2, self.simLabel_sketch)
            self.GM_shape, self.expGM_shape = self.calculateGroundMetricContrastive(self.output_shape_fea_1, self.output_shape_fea_2, self.simLabel_shape)
            self.GM_cross_1, self.expGM_cross_1 = self.calculateGroundMetricContrastive(self.output_sketch_fea_1, self.output_shape_fea_1, self.simLabel_cross_1)
            self.GM_cross_2, self.expGM_cross_2 = self.calculateGroundMetricContrastive(self.output_sketch_fea_2, self.output_shape_fea_2, self.simLabel_cross_2)




            # create loss

            self.loss = tf.reduce_mean(tf.add_n([self.GM_sketch, self.GM_shape, self.GM_cross_1, self.GM_cross_2]), name='loss')
            # distance used for retrieval


            # self.loss_summary_sketch = tf.summary.scalar('sketch_loss', self.loss_sketch)
            # self.loss_summary_shape = tf.summary.scalar('shape_loss', self.loss_shape)
            # self.loss_summary_cross_1 = tf.summary.scalar('cross_loss_1', self.loss_cross_1)
            # self.loss_summary_cross_2 = tf.summary.scalar('cross_loss_2', self.loss_cross_2)
            self.loss_summary = tf.summary.scalar('loss', self.loss)




        var_list = tf.trainable_variables()
        self.gm_var_list = [var for var in var_list if ('sketch' in var.name or 'shape' in var.name)]
        for var in self.gm_var_list:
            print(var.name)


    def ckpt_status(self):
        print("[*] Reading checkpoint ...")
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.model_checkpoint_path = ckpt.model_checkpoint_path
            return True
        else:
            return None

    def train(self):
        #self.gm_optim = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(self.loss, var_list=self.gm_var_list)
        #global_step = tf.Variable(0, trainable=False)
        #learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 10000, 0.7)
        #self.gm_optim = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum).minimize(self.loss, var_list=self.gm_var_list)
        self.gm_optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss, var_list=self.gm_var_list)
        # self.shape_optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss_shape, var_list=self.gm_var_list)
        # self.sketch_optim = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum).minimize(self.loss_sketch, var_list=self.gm_var_list)
        #self.gm_optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.gm_var_list)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        start_time = time.time()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter(self.logdir, sess.graph)
            sess.run(init)

            if self.ckpt_status():
                print("[*] Load SUCCESS")
                saver.restore(sess, self.model_checkpoint_path)
            else:
                print("[*] Load failed")

            for iter in range(self.maxiter):

                sketch_fea_1, sketch_label_1 = self.nextBatch(self.batch_size, 'sketch_train')

                shape_fea_1, shape_label_1 = self.nextBatch(self.batch_size, 'shape')

                sketch_fea_2, sketch_label_2 = self.nextBatch(self.batch_size, 'sketch_train')
                shape_fea_2, shape_label_2 = self.nextBatch(self.batch_size, 'shape')

                if self.lossType == 'contrastiveLoss':
                    _, loss_, loss_sum_, sketch_fea, shape_fea = sess.run([self.gm_optim, self.loss, self.loss_summary, self.output_sketch_fea_1, self.output_shape_fea_1], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                            })

                elif self.lossType == 'doubleContrastiveLoss':
                    _, loss_, loss_sum_, sketch_fea, shape_fea, M_eval_1 = sess.run([self.gm_optim, self.loss, self.loss_summary, self.sketch_1, self.shape_1, self.groundMetric_1], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                            })

                elif self.lossType == 'weightedContrastiveLoss':
                    M_sketch, M_shape, M_cross_1, M_cross_2 = sess.run([self.expGM_sketch, self.expGM_shape, self.expGM_cross_1, self.expGM_cross_2], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                            })


                    # sinkhorn iteration
                    self.shape_u_reset.eval()
                    self.sketch_u_reset.eval()
                    self.cross_1_u_reset.eval()
                    self.cross_2_u_reset.eval()

                    for sinhorn_ter in range(20):

                        self.sketch_v_assign_op.eval(feed_dict={self.expGM_sketch:M_sketch})
                        self.sketch_u_assign_op.eval(feed_dict={self.expGM_sketch:M_sketch})


                        self.shape_v_assign_op.eval(feed_dict={self.expGM_shape:M_shape})
                        self.shape_u_assign_op.eval(feed_dict={self.expGM_shape:M_shape})

                        self.cross_1_v_assign_op.eval(feed_dict={self.expGM_cross_1:M_cross_1})
                        self.cross_1_u_assign_op.eval(feed_dict={self.expGM_cross_1:M_cross_1})

                        self.cross_2_v_assign_op.eval(feed_dict={self.expGM_cross_2:M_cross_2})
                        self.cross_2_u_assign_op.eval(feed_dict={self.expGM_cross_2:M_cross_2})

                    _, loss_, loss_sum_, sketch_fea, shape_fea = sess.run([self.gm_optim, self.loss, self.loss_summary, self.output_sketch_fea_1, self.output_shape_fea_1], feed_dict={
                            self.input_sketch_fea_1: sketch_fea_1,
                            self.input_sketch_label_1: sketch_label_1,
                            self.input_sketch_fea_2: sketch_fea_2,
                            self.input_sketch_label_2: sketch_label_2,
                            self.input_shape_fea_1: shape_fea_1,
                            self.input_shape_label_1: shape_label_1,
                            self.input_shape_fea_2: shape_fea_2,
                            self.input_shape_label_2: shape_label_2
                            })


                writer.add_summary(loss_sum_, iter)
                # reset u values
                if iter % 500  == 0:       # every 10 batches, update sinkhorn
                    print("Iteration: [%5d] [total number of examples: %5d] time: %4.4f, loss: %.8f" % (iter, self.shape_num, time.time() - start_time, loss_))

                # This is for debuging, not saving the checkpoint
                if iter % 5000 == 0:
                    saver.save(sess, os.path.join(self.ckpt_dir, self.ckpt_name), global_step=iter)
                # self.evaluation_online(sess)

    def evaluation_online(self, sess):
        self.getLabel()

        # initialize all the array to evaluation
        testSketchNumber = len(self.sketch_test_label)
        trainShapeNumber = len(self.shape_label)

        distanceMatrix = np.zeros((testSketchNumber, trainShapeNumber))
        sketchMatrix = np.zeros((testSketchNumber, 100))
        shapeMatrix = np.zeros((trainShapeNumber, 100))

        print("First Load all the data")
        start_time = time.time()
        tmp = np.zeros((self.batch_size, self.num_views, self.inputFeaSize))

        if testSketchNumber % self.batch_size == 0:
            rangeNumber = testSketchNumber
        else:
            rangeNumber = testSketchNumber + 1
        for i in range(0, rangeNumber, self.batch_size):
            print("Loading the {:5d}-th sketch".format(i))
            for j in range(self.batch_size):
                for k in range(self.num_views):
                    filePath = self.sketch_test_data[i+j][k].split(' ')[0]
                    tmp[j,k] = np.loadtxt(filePath)
            tmp_out = sess.run(self.sketch_out, feed_dict={self.input_sketch_fea: tmp})
            tmp_out = np.reshape(tmp_out, (self.batch_size, self.num_views, 100))
            tmp_out = np.amax(tmp_out, axis=1)      # max-pooling across views
            tmp_out = np.reshape(tmp_out, [self.batch_size, 100])

            if i + self.batch_size <= testSketchNumber:
                sketchMatrix[i:i+self.batch_size] = tmp_out
            else:
                sketchMatrix[i:] = tmp_out[:(i+self.batch_size)%testSketchNumber]

        print("Time for Loading sketch is {}\n".format(time.time() - start_time))

        start_time = time.time()

        if trainShapeNumber % self.batch_size == 0:
            rangeNumber = trainShapeNumber
        else:
            rangeNumber = trainShapeNumber + 1
        for i in range(0, rangeNumber, self.batch_size):
            print("Loading the {:5d}-th shape".format(i))
            for j in range(self.batch_size):
                for k in range(self.num_views):
                    filePath = self.shape_data[i+j][k].split(' ')[0]
                    tmp[j,k] = np.loadtxt(filePath)
            tmp_out = sess.run(self.shape_out, feed_dict={self.input_shape_fea: tmp})
            tmp_out = np.reshape(tmp_out, (self.batch_size, self.num_views, 100))
            tmp_out = np.amax(tmp_out, axis=1)      # max-pooling across views
            tmp_out = np.reshape(tmp_out, [self.batch_size, 100])
            if i + self.batch_size <= trainShapeNumber:
                shapeMatrix[i:i+self.batch_size] = tmp_out
            else:
                shapeMatrix[i:] = tmp_out[:(i+self.batch_size)%trainShapeNumber]

        print("Time for Loading shape is {}\n".format(time.time() - start_time))

        distM = distance.cdist(sketchMatrix, shapeMatrix)

        print("calculating distance is finished")

        model_label = np.array(self.shape_label).astype(int)
        test_label = np.array(self.sketch_test_label).astype(int)
        C_depths = self.retrievalParamSP()
        C_depths = C_depths.astype(int)
        nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distanceMatrix, model_label, test_label, testMode=1)

        print 'The NN is %5f' % (nn_av)
        print 'The FT is %5f' % (ft_av)
        print 'The ST is %5f' % (st_av)
        print 'The DCG is %5f' % (dcg_av)
        print 'The E is %5f' % (e_av)
        print 'The MAP is %5f' % (map_)



    def evaluation(self):
        init = tf.global_variables_initializer()        # init all variables
        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            sess.run(init)
            if self.ckpt_status():
                print("[*] Load SUCCESS")
                saver.restore(sess, self.model_checkpoint_path)
            else:
                print("[*] Load failed")


            distanceMatrix = np.zeros((self.sketch_test_num, self.shape_num))
            sketchMatrix = np.zeros((self.sketch_test_num, self.outputFeaSize))
            shapeMatrix = np.zeros((self.shape_num, self.outputFeaSize))

            start_time = time.time()
            print(self.sketch_test_num)
            print(self.sketchTestFeaset.shape)
            print(self.batch_size)
            for i in range(0, self.sketch_test_num, self.batch_size):

                if i + self.batch_size <= self.sketch_test_num:
                    tmp_in = self.sketchTestFeaset[i:i+self.batch_size]
                else:
                    tmp_in = np.zeros((self.batch_size, self.outputFeaSize))        # empty matrix
                    tmp_in[:self.sketch_test_num-i] = self.sketchTestFeaset[i:]     # Get the last batch (# is less than batch size)

                print(tmp_in.shape)
                tmp_out = sess.run(self.output_sketch_fea_1, feed_dict={self.input_sketch_fea_1: tmp_in})

                if i + self.batch_size <= self.sketch_test_num:
                    sketchMatrix[i:i+self.batch_size] = tmp_out
                else:
                    sketchMatrix[i:] = tmp_out[:self.sketch_test_num-i]     # Get the remaining data of the last batch

            print("Time for Loading sketch is {}\n".format(time.time() - start_time))

            start_time = time.time()
            print(self.shapeFeaset.shape)
            for i in range(0, self.shape_num, self.batch_size):
                if i + self.batch_size <= self.shape_num:
                    tmp_in = self.shapeFeaset[i:i+self.batch_size]
                else:
                    tmp_in = np.zeros((self.batch_size, self.outputFeaSize))        # empty matrix
                    tmp_in[:self.shape_num-i] = self.shapeFeaset[i:]     # Get the last batch (# is less than batch size)

                tmp_out = sess.run(self.output_shape_fea_1, feed_dict={self.input_shape_fea_1: tmp_in})

                if i + self.batch_size <= self.shape_num:
                    shapeMatrix[i:i+self.batch_size] = tmp_out
                else:
                    shapeMatrix[i:] = tmp_out[:self.shape_num-i]     # Get the remaining data of the last batc

            distanceMatrix = distance.cdist(sketchMatrix, shapeMatrix)

            model_label = (self.shapeLabelset).astype(int)
            test_label = (self.sketchTestLabelset).astype(int)
            C_depths = self.retrievalParamSP()
            C_depths = C_depths.astype(int)

            print("Retrieval evaluation")
            nn_av, ft_av, st_av, dcg_av, e_av, map_, p_points, pre, rec = RetrievalEvaluation(C_depths, distanceMatrix, model_label, test_label, testMode=1)

            print 'The NN is %5f' % (nn_av)
            print 'The FT is %5f' % (ft_av)
            print 'The ST is %5f' % (st_av)
            print 'The DCG is %5f' % (dcg_av)
            print 'The E is %5f' % (e_av)
            print 'The MAP is %5f' % (map_)