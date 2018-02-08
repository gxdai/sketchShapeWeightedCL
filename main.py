import argparse
import os 
import scipy.misc 
import numpy as np 
import os 
from model import model
import tensorflow as tf

parser = argparse.ArgumentParser(description='This is the model for training weighted contrastive loss')
parser.add_argument('--lamb', dest='lamb', type=float, default=100., help='parameter for sinkhorn iteration')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', type=str, default='./checkpoint', help='directory for saving checkpoint')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=30, help='# images in batch')
parser.add_argument('--margin', type=float, default=10, help='The margin of contrastive loss')
parser.add_argument('--sketch_train_list', type=str, default='./sketch_train.txt', help='The training list file')
parser.add_argument('--sketch_test_list', type=str, default='./sketch_test.txt', help='The testing list file')
parser.add_argument('--shape_list', type=str, default='./shape.txt', help='The shape list file')
parser.add_argument('--num_views', type=int, default=20, help='The total number of views')
parser.add_argument('--num_views_sketch', type=int, default=20, help='The total number of views for sketches')
parser.add_argument('--num_views_shape', type=int, default=20, help='The total number of views for shape')
parser.add_argument('--class_num', type=int, default=90, help='the total number of class')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, evaluation')
parser.add_argument('--logdir', dest='logdir', default='./logs', help='name of the dataset')
parser.add_argument('--maxiter', dest='maxiter', type=int, default=100000, help='maximum number of iterations') 
parser.add_argument('--inputFeaSize', dest='inputFeaSize', type=int, default=4096, help='The dimensions of input features') 
parser.add_argument('--outputFeaSize', dest='outputFeaSize', type=int, default=100, help='The dimensions of input features') 
parser.add_argument('--lossType', dest='lossType', type=str, default='contrastiveLoss', help='name of the dataset')
parser.add_argument('--activationType', dest='activationType', type=str, default='relu', help='name of the dataset')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default='0.0005', help='learning rate')
parser.add_argument('--normFlag', dest='normFlag', type=int, default=0, help='whether to normalize the input feature')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.5, help='momentum term of Gradient')
args = parser.parse_args()

def main(_):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        print(args.ckpt_dir)
    wasserteinModel = model(lamb=args.lamb, ckpt_dir=args.ckpt_dir, batch_size=args.batch_size, margin=args.margin,
            sketch_train_list=args.sketch_train_list, sketch_test_list=args.sketch_test_list, shape_list=args.shape_list, 
            num_views_shape=args.num_views_shape, learning_rate=args.learning_rate, momentum=args.momentum,
            class_num=args.class_num, normFlag=args.normFlag, logdir=args.logdir, lossType=args.lossType, activationType=args.activationType, 
            phase=args.phase, inputFeaSize=args.inputFeaSize, outputFeaSize=args.outputFeaSize, maxiter=args.maxiter)

    if args.phase == 'train':
        wasserteinModel.train()
    elif args.phase == 'test':
        wasserteinModel.test(args)
    elif args.phase == 'evaluation':
        wasserteinModel.evaluation()

if __name__ == '__main__':
    tf.app.run()
