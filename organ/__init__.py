from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from builtins import range
from collections import OrderedDict, defaultdict
import numpy as np
import random
import dill as pickle
from organ.data_loaders import Gen_Dataloader, Dis_Dataloader
from organ.generator import Generator
from organ.wgenerator import WGenerator
from organ.rollout import Rollout
from organ.discriminator import Discriminator
from organ.wdiscriminator import WDiscriminator
from organ.prior_classifier import prior_classifier
from organ.classify_rollout import ClassifyRollout
#from organ.discriminator import WDiscriminator as Discriminator

from tensorflow import logging
from rdkit import rdBase
import pandas as pd
from tqdm import tqdm, trange
import organ.mol_metrics
import organ.music_metrics
from collections import Counter


class ORGAN(object):
    """Main class, where every interaction between the user
    and the backend is performed.
    """

    def __init__(self, name, metrics_module, params={},
                 verbose=True):
        """Parameter initialization.

        Arguments
        -----------

            - name. String which will be used to identify the
            model in any folders or files created.

            - metrics_module. String identifying the module containing
            the metrics.

            - params. Optional. Dictionary containing the parameters
            that the user whishes to specify.

            - verbose. Boolean specifying whether output must be
            produced in-line.

        """

        self.verbose = verbose

        # Set minimum verbosity for RDKit, Keras and TF backends
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        logging.set_verbosity(logging.INFO)
        rdBase.DisableLog('rdApp.error')

        # GPU configuration
        if 'GPU' in params:
            self.GPU = params['GPU']
        else:
            self.GPU = True  # Default: use GPU
        
        # Set GPU configuration
        self.config = tf.ConfigProto()
        if self.GPU:
            # Get available GPU devices
            try:
                from tensorflow.python.client import device_lib
                local_devices = device_lib.list_local_devices()
                gpus = [x for x in local_devices if x.device_type == 'GPU']
                if gpus:
                    if self.verbose:
                        print(f"Using GPU: {len(gpus)} available")
                    self.config.gpu_options.allow_growth = True  # Dynamic memory allocation
                    self.config.gpu_options.per_process_gpu_memory_fraction = 0.9  # Use up to 90% GPU memory
                else:
                    if self.verbose:
                        print("No GPU found, using CPU")
                    self.GPU = False
                    self.config.device_count['GPU'] = 0
            except:
                if self.verbose:
                    print("Failed to get GPU info, using CPU")
                self.GPU = False
                self.config.device_count['GPU'] = 0
        else:
            self.config.device_count['GPU'] = 0  # Disable GPU

        # Set parameters
        self.PREFIX = name
        if 'WGAN' in params:
            self.WGAN = params['WGAN']
        else:
            self.WGAN = False

        if 'PRETRAIN_GEN_EPOCHS' in params:
            self.PRETRAIN_GEN_EPOCHS = params['PRETRAIN_GEN_EPOCHS']
        else:
            self.PRETRAIN_GEN_EPOCHS = 240

        if 'PRETRAIN_DIS_EPOCHS' in params:
            self.PRETRAIN_DIS_EPOCHS = params['PRETRAIN_DIS_EPOCHS']
        else:
            self.PRETRAIN_DIS_EPOCHS = 50

        if 'GEN_ITERATIONS' in params:
            self.GEN_ITERATIONS = params['GEN_ITERATIONS']
        else:
            self.GEN_ITERATIONS = 2

        if 'GEN_BATCH_SIZE' in params:
            self.GEN_BATCH_SIZE = params['GEN_BATCH_SIZE']
        else:
            self.GEN_BATCH_SIZE = 64

        if 'SEED' in params:
            self.SEED = params['SEED']
        else:
            self.SEED = None
        random.seed(self.SEED)
        np.random.seed(self.SEED)

        if 'DIS_BATCH_SIZE' in params:
            self.DIS_BATCH_SIZE = params['DIS_BATCH_SIZE']
        else:
            self.DIS_BATCH_SIZE = 64

        if 'DIS_EPOCHS' in params:
            self.DIS_EPOCHS = params['DIS_EPOCHS']
        else:
            self.DIS_EPOCHS = 3

        if 'EPOCH_SAVES' in params:
            self.EPOCH_SAVES = params['EPOCH_SAVES']
        else:
            self.EPOCH_SAVES = 20

        if 'CHK_PATH' in params:
            self.CHK_PATH = params['CHK_PATH']
        else:
            self.CHK_PATH = os.path.join(
                os.getcwd(), 'checkpoints/{}'.format(self.PREFIX))

        if 'GEN_EMB_DIM' in params:
            self.GEN_EMB_DIM = params['GEN_EMB_DIM']
        else:
            self.GEN_EMB_DIM = 32

        if 'GEN_HIDDEN_DIM' in params:
            self.GEN_HIDDEN_DIM = params['GEN_HIDDEN_DIM']
        else:
            self.GEN_HIDDEN_DIM = 32

        if 'START_TOKEN' in params:
            self.START_TOKEN = params['START_TOKEN']
        else:
            self.START_TOKEN = 0

        if 'SAMPLE_NUM' in params:
            self.SAMPLE_NUM = params['SAMPLE_NUM']
        else:
            self.SAMPLE_NUM = 6400

        if 'CLASS_NUM' in params:
            self.CLASS_NUM = params['CLASS_NUM']
        else:
            self.CLASS_NUM = 1

        if 'BIG_SAMPLE_NUM' in params:
            self.BIG_SAMPLE_NUM = params['BIG_SAMPLE_NUM']
        else:
            self.BIG_SAMPLE_NUM = self.SAMPLE_NUM * 5

        if 'LAMBDA' in params:
            self.LAMBDA = params['LAMBDA']
        else:
            self.LAMBDA = 0.5

        # In case this parameter is not specified by the user,
        # it will be determined later, in the training set
        # loading.
        if 'MAX_LENGTH' in params:
            self.MAX_LENGTH = params['MAX_LENGTH']

        if 'DIS_EMB_DIM' in params:
            self.DIS_EMB_DIM = params['DIS_EMB_DIM']
        else:
            self.DIS_EMB_DIM = 64

        if 'DIS_FILTER_SIZES' in params:
            self.DIS_FILTER_SIZES = params['DIS_FILTER_SIZES']
        else:
            self.DIS_FILTER_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

        if 'DIS_NUM_FILTERS' in params:
            self.DIS_NUM_FILTERS = params['DIS_NUM_FILTERS']
        else:
            self.DIS_NUM_FILTERS = [100, 200, 200, 200, 200, 100,
                                    100, 100, 100, 100, 160, 160]

        if 'DIS_DROPOUT' in params:
            self.DIS_DROPOUT = params['DIS_DROPOUT']
        else:
            self.DIS_DROPOUT = 0.75
        if 'DIS_GRAD_CLIP' in params:
            self.DIS_GRAD_CLIP = params['DIS_GRAD_CLIP']
        else:
            self.DIS_GRAD_CLIP = 1.0

        if 'WGAN_REG_LAMBDA' in params:
            self.WGAN_REG_LAMBDA = params['WGAN_REG_LAMBDA']
        else:
            self.WGAN_REG_LAMBDA = 1.0

        if 'DIS_L2REG' in params:
            self.DIS_L2REG = params['DIS_L2REG']
        else:
            self.DIS_L2REG = 0.2

        if 'TBOARD_LOG' in params:
            print('Tensorboard functionality')

        global mm
        if metrics_module == 'mol_metrics':
            mm = mol_metrics
        elif metrics_module == 'music_metrics':
            mm = music_metrics
        else:
            raise ValueError('Undefined metrics')

        self.AV_METRICS = mm.get_metrics()
        self.LOADINGS = mm.metrics_loading()

        self.PRETRAINED = False
        self.SESS_LOADED = False
        self.USERDEF_METRIC = False
        self.PRIOR_CLASSIFIER = False
        self.ORGAN_TRAINED = False
        
    def load_training_set(self, file):
        """Specifies a training set for the model. It also finishes
        the model set up, as some of the internal parameters require
        knowledge of the vocabulary.

        Arguments
        -----------

            - file. String pointing to the dataset file.

        """

        # Load training set
        self.train_samples = mm.load_train_data(file)
        self.molecules, _ = zip(*self.train_samples)

        # Process and create vocabulary
        self.char_dict, self.ord_dict = mm.build_vocab(self.molecules, class_num=self.CLASS_NUM)
        self.NUM_EMB = len(self.char_dict)
        self.PAD_CHAR = self.ord_dict[self.NUM_EMB - 1]
        self.PAD_NUM = self.char_dict[self.PAD_CHAR]
        self.DATA_LENGTH = max(map(len, self.molecules))
        print('Vocabulary:')
        print(list(self.char_dict.keys()))
        # If MAX_LENGTH has not been specified by the user, it
        # will be set as 1.5 times the maximum length in the
        # trining set.
        if not hasattr(self, 'MAX_LENGTH'):
            self.MAX_LENGTH = int(len(max(self.molecules, key=len)) * 1.5)

        # Encode samples
        to_use = [sample for sample in self.train_samples
                  if mm.verified_and_below(sample[0], self.MAX_LENGTH)]
        molecules_to_use, label_to_use = zip(*to_use)
        positive_molecules = [mm.encode(sam,
                            self.MAX_LENGTH,
                            self.char_dict) for sam in molecules_to_use]
        self.positive_samples = [list(item) for item in zip(positive_molecules, label_to_use)]
        self.POSITIVE_NUM = len(self.positive_samples)
        self.TYPE_NUM = Counter([sam[1] for sam in to_use])

        # Print information
        if self.verbose:

            print('\nPARAMETERS INFORMATION')
            print('============================\n')
            print('Model name               :   {}'.format(self.PREFIX))
            print('Training set size        :   {} points'.format(
                len(self.train_samples)))
            print('Max data length          :   {}'.format(self.MAX_LENGTH))
            lens = [len(s[0]) for s in to_use]
            print('Avg Length to use is     :   {:2.2f} ({:2.2f}) [{:d},{:d}]'.format(
                np.mean(lens), np.std(lens), np.min(lens), np.max(lens)))
            print('Num valid data points is :   {}'.format(
                self.POSITIVE_NUM))
            print('Num different samples is :   {}'.format(
                self.TYPE_NUM))
            print('Size of alphabet is      :   {}'.format(self.NUM_EMB))
            print('')

            params = ['PRETRAIN_GEN_EPOCHS', 'PRETRAIN_DIS_EPOCHS',
                      'GEN_ITERATIONS', 'GEN_BATCH_SIZE', 'SEED',
                      'DIS_BATCH_SIZE', 'DIS_EPOCHS', 'EPOCH_SAVES',
                      'CHK_PATH', 'GEN_EMB_DIM', 'GEN_HIDDEN_DIM',
                      'START_TOKEN', 'SAMPLE_NUM', 'BIG_SAMPLE_NUM',
                      'LAMBDA', 'MAX_LENGTH', 'DIS_EMB_DIM',
                      'DIS_FILTER_SIZES', 'DIS_NUM_FILTERS',
                      'DIS_DROPOUT', 'DIS_L2REG']

            for param in params:
                string = param + ' ' * (25 - len(param))
                value = getattr(self, param)
                print('{}:   {}'.format(string, value))

        # Set model
        self.gen_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        self.dis_loader = Dis_Dataloader()
        self.mle_loader = Gen_Dataloader(self.GEN_BATCH_SIZE)
        if self.WGAN:
            self.generator = WGenerator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                                        self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                                        self.MAX_LENGTH, self.START_TOKEN)
            self.discriminator = WDiscriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG,
                wgan_reg_lambda=self.WGAN_REG_LAMBDA,
                grad_clip=self.DIS_GRAD_CLIP)
        else:
            self.generator = Generator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                                       self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                                       self.MAX_LENGTH, self.START_TOKEN)
            self.discriminator = Discriminator(
                sequence_length=self.MAX_LENGTH,
                num_classes=2,
                vocab_size=self.NUM_EMB,
                embedding_size=self.DIS_EMB_DIM,
                filter_sizes=self.DIS_FILTER_SIZES,
                num_filters=self.DIS_NUM_FILTERS,
                l2_reg_lambda=self.DIS_L2REG,
                grad_clip=self.DIS_GRAD_CLIP)

        # run tensorflow
        self.sess = tf.Session(config=self.config)

        #self.tb_write = tf.summary.FileWriter(self.log_dir)

    def define_metric(self, name, metric, load_metric=lambda *args: None,
                      pre_batch=False, pre_metric=lambda *args: None):
        """Sets up a new metric and generates a .pkl file in
        the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metric. Function taking as argument a sequence
            and returning a float value.

            - load_metric. Optional. Preprocessing needed
            at the beginning of the code.

            - pre_batch. Optional. Boolean specifying whether
            there is any preprocessing when the metric is applied
            to a batch of sequences. False by default.

            - pre_metric. Optional. Preprocessing operations
            for the metric. Will be ignored if pre_batch is False.

        Note
        -----------

            For combinations of already existing metrics, check
            the define_metric_as_combination method.

        """

        if pre_batch:
            def batch_metric(smiles, train_smiles=None):
                psmiles = pre_metric()
                vals = [mm.apply_to_valid(s, metric) for s in psmiles]
                return vals
        else:
            def batch_metric(smiles, train_smiles=None):
                vals = [mm.apply_to_valid(s, metric) for s in smiles]
                return vals

        self.AV_METRICS[name] = batch_metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [batch_metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def define_metric_as_combination(self, name, metrics, ponderations):
        """Sets up a metric made from a combination of
        previously existing metrics. Also generates a
        metric .pkl file in the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metrics. List containing the name identifiers
            of every metric in the list

            - ponderations. List of ponderation coefficients
            for every metric in the previous list.

        """

        funs = [self.AV_METRICS[metric] for metric in metrics]
        funs_load = [self.LOADINGS[metric] for metric in metrics]

        def metric(smiles, train_smiles=None, **kwargs):
            vals = np.zeros(len(smiles))
            for fun, c in zip(funs, ponderations):
                vals += c * np.asarray(fun(smiles))
            return vals

        def load_metric():
            return [fun() for fun in funs_load if fun() is not None]

        self.AV_METRICS[name] = metric
        self.LOADINGS[name] = load_metric

        if self.verbose:
            print('Defined metric {}'.format(name))

        nmetric = [metric, load_metric]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(nmetric, f)

    def define_metric_as_remap(self, name, metric, remapping):
        """Sets up a metric made from a remapping of a
        previously existing metric. Also generates a .pkl
        metric file in the data/ directory.

        Arguments
        -----------

            - name. String used to identify the metric.

            - metric. String identifying the previous metric.

            - remapping. Remap function.

        """

        pmetric = self.AV_METRICS[metric]

        def nmetric(smiles, train_smiles=None, **kwargs):
            vals = pmetric(smiles, train_smiles, **kwargs)
            return remapping(vals)

        self.AV_METRICS[name] = nmetric
        self.LOADINGS[name] = self.LOADINGS[metric]

        if self.verbose:
            print('Defined metric {}'.format(name))

        metric = [nmetric, self.LOADINGS[metric]]
        with open('../data/{}.pkl'.format(name), 'wb') as f:
            pickle.dump(metric, f)

    def load_prev_user_metric(self, name, file=None):
        """Loads a metric that the user has previously designed.

        Arguments.
        -----------

            - name. String used to identify the metric.

            - file. String pointing to the .pkl file. Will use
            ../data/name.pkl by default.

        """
        if file is None:
            file = '../data/{}.pkl'.format(name)
        pkl = open(file, 'rb')
        data = pickle.load(pkl)
        self.AV_METRICS[name] = data[0]
        self.LOADINGS[name] = data[1]
        if self.verbose:
            print('Loaded metric {}'.format(name))

    def set_training_program(self, metrics=None, steps=None):
        """Sets a program of metrics and epochs
        for training the model and generating molecules.

        Arguments
        -----------

            - metrics. List of metrics. Each element represents
            the metric used with a particular set of epochs. Its
            length must coincide with the steps list.

            - steps. List of epoch sets. Each element represents
            the number of epochs for which a given metric will
            be used. Its length must coincide with the steps list.

        Note
        -----------

            The program will crash if both lists have different
            lengths.

        """

        # Raise error if the lengths do not match
        if len(metrics) != len(steps):
            return ValueError('Unmatching lengths in training program.')

        # Set important parameters
        self.TOTAL_BATCH = np.sum(np.asarray(steps))
        self.METRICS = metrics

        # Build the 'educative program'
        self.EDUCATION = {}
        i = 0
        for j, stage in enumerate(steps):
            for _ in range(stage):
                self.EDUCATION[i] = metrics[j]
                i += 1

    def load_metrics(self):
        """Loads the metrics."""

        # Get the list of used metrics
        met = list(set(self.METRICS))

        # Execute the metrics loading
        self.kwargs = {}
        for m in met:
            load_fun = self.LOADINGS[m]
            args = load_fun()
            if args is not None:
                if isinstance(args, tuple):
                    self.kwargs[m] = {args[0]: args[1]}
                elif isinstance(args, list):
                    fun_args = {}
                    for arg in args:
                        fun_args[arg[0]] = arg[1]
                    self.kwargs[m] = fun_args
            else:
                self.kwargs[m] = None

    def load_prev_pretraining(self, ckpt=None):
        """
        Loads a previous pretraining.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name_pretrain/pretrain_ckpt' is assumed.

        Note
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files, like in the following ls:

                checkpoint
                pretrain_ckpt.data-00000-of-00001
                pretrain_ckpt.index
                pretrain_ckpt.meta

            In this case, ckpt = 'pretrain_ckpt'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # Generate TF saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}_pretrain'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        # Load from checkpoint
        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Pretrain loaded from previous checkpoint {}'.format(ckpt))
            self.PRETRAINED = True
        else:
            print('\t* No pre-training data found as {:s}.'.format(ckpt))

    def load_prev_training(self, ckpt=None):
        """
        Loads a previous trained model.

        Arguments
        -----------

            - ckpt. String pointing to the ckpt file. By default,
            'checkpoints/name/pretrain_ckpt' is assumed.

        Note 1
        -----------

            The models are stored by the Tensorflow API backend. This
            will generate various files. An example ls:

                checkpoint
                validity_model_0.ckpt.data-00000-of-00001
                validity_model_0.ckpt.index
                validity_model_0.ckpt.meta
                validity_model_100.ckpt.data-00000-of-00001
                validity_model_100.ckpt.index
                validity_model_100.ckpt.meta
                validity_model_120.ckpt.data-00000-of-00001
                validity_model_120.ckpt.index
                validity_model_120.ckpt.meta
                validity_model_140.ckpt.data-00000-of-00001
                validity_model_140.ckpt.index
                validity_model_140.ckpt.meta

                    ...

                validity_model_final.ckpt.data-00000-of-00001
                validity_model_final.ckpt.index
                validity_model_final.ckpt.meta

            Possible ckpt values are 'validity_model_0', 'validity_model_140'
            or 'validity_model_final'.

        Note 2
        -----------

            Due to its structure, ORGANIC is very dependent on its
            hyperparameters (for example, MAX_LENGTH defines the
            embedding). Most of the errors with this function are
            related to parameter mismatching.

        """

        # If there is no Rollout, add it
        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        # Generate TF Saver
        saver = tf.train.Saver()

        # Set default checkpoint
        if ckpt is None:
            ckpt_dir = 'checkpoints/{}'.format(self.PREFIX)
            if not os.path.exists(ckpt_dir):
                print('No pretraining data was found')
                return
            ckpt = os.path.join(ckpt_dir, 'pretrain_ckpt')

        if os.path.isfile(ckpt + '.meta'):
            saver.restore(self.sess, ckpt)
            print('Training loaded from previous checkpoint {}'.format(ckpt))
            self.SESS_LOADED = True
            self.ORGAN_TRAINED = True
        else:
            print('\t* No training checkpoint found as {:s}.'.format(ckpt))

    def pretrain(self):
        """Pretrains generator and discriminator."""

        self.gen_loader.create_batches(self.positive_samples)
        # results = OrderedDict({'exp_name': self.PREFIX})

        if self.verbose:
            print('\nPRETRAINING')
            print('============================\n')
            print('GENERATOR PRETRAINING')

        t_bar = trange(self.PRETRAIN_GEN_EPOCHS)
        for epoch in t_bar:
            supervised_g_losses = []
            self.gen_loader.reset_pointer()
            for it in range(self.gen_loader.num_batch):
                batch = self.gen_loader.next_batch()
                x, class_label = zip(*batch)
                _, g_loss, g_pred = self.generator.pretrain_step(self.sess,
                                                                 x)
                supervised_g_losses.append(g_loss)
            # print results
            mean_g_loss = np.mean(supervised_g_losses)
            t_bar.set_postfix(G_loss=mean_g_loss)

        samples = self.generate_samples(self.SAMPLE_NUM)
        self.mle_loader.create_batches(samples)

        if self.LAMBDA != 0:

            if self.verbose:
                print('\nDISCRIMINATOR PRETRAINING')
            t_bar = trange(self.PRETRAIN_DIS_EPOCHS)
            for i in t_bar:
                negative_samples = self.generate_samples(self.POSITIVE_NUM)
                dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                    self.positive_samples, negative_samples)
                dis_batches = self.dis_loader.batch_iter(
                    zip(dis_x_train, dis_y_train), self.DIS_BATCH_SIZE,
                    self.PRETRAIN_DIS_EPOCHS)
                supervised_d_losses = []
                for batch in dis_batches:
                    x_batch, y_batch = zip(*batch)
                    x, x_label = zip(*x_batch)
                    # x_batch.size = (batch_size, sequence_length), y_batch.size = (batch_size, 2)
                    _, d_loss, _, _, _ = self.discriminator.train(
                        self.sess, x, y_batch, self.DIS_DROPOUT)

                    supervised_d_losses.append(d_loss)
                # print results
                mean_d_loss = np.mean(supervised_d_losses)
                t_bar.set_postfix(D_loss=mean_d_loss)

        self.PRETRAINED = True
        return

    def generate_samples(self, num, label_input=False, target_class=None):
        """Generates molecules.

        Arguments
        -----------
            - num. Integer 表示要生成的分子数量
            - label_input. Boolean 是否将标签作为输入

        """
        generated_samples = []

        for _ in range(int(num / self.GEN_BATCH_SIZE)):
            for class_label in range(0, self.CLASS_NUM):
                samples = self.generator.generate(self.sess)
                # 将生成的样本和对应的标签组合
                for i in range(self.GEN_BATCH_SIZE):
                    generated_samples.append([samples[i].tolist(), class_label])

        return generated_samples

    def report_rewards(self, rewards, metric):
        print('~~~~~~~~~~~~~~~~~~~~~~~~\n')
        print('Reward: {}  (lambda={:.2f})'.format(metric, self.LAMBDA))
        #np.set_printoptions(precision=3, suppress=True)
        mean_r, std_r = np.mean(rewards), np.std(rewards)
        min_r, max_r = np.min(rewards), np.max(rewards)
        print('Stats: {:.3f} ({:.3f}) [{:3f},{:.3f}]'.format(
            mean_r, std_r, min_r, max_r))
        non_neg = rewards[rewards > 0.01]
        if len(non_neg) > 0:
            mean_r, std_r = np.mean(non_neg), np.std(non_neg)
            min_r, max_r = np.min(non_neg), np.max(non_neg)
            print('Valid: {:.3f} ({:.3f}) [{:3f},{:.3f}]'.format(
                mean_r, std_r, min_r, max_r))
        #np.set_printoptions(precision=8, suppress=False)
        return

    def organ_train(self, ckpt_dir='checkpoints/'):
        """Trains the model. If necessary, also includes pretraining."""

        if not self.PRETRAINED and not self.SESS_LOADED:

            self.sess.run(tf.global_variables_initializer())
            self.pretrain()

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_file = os.path.join(ckpt_dir,
                                     '{}_pretrain_ckpt'.format(self.PREFIX))
            saver = tf.train.Saver()
            path = saver.save(self.sess, ckpt_file)
            if self.verbose:
                print('Pretrain saved at {}'.format(path))

        if not hasattr(self, 'rollout'):
            self.rollout = Rollout(self.generator, 0.8, self.PAD_NUM)

        if self.verbose:
            print('\nSTARTING ORGAN TRAINING')
            print('============================\n')

        results_rows = []
        losses = defaultdict(list)
        for nbatch in tqdm(range(self.TOTAL_BATCH)):

            results = OrderedDict({'exp_name': self.PREFIX})
            metric = self.EDUCATION[nbatch]

            if metric in self.AV_METRICS.keys():
                reward_func = self.AV_METRICS[metric]
            else:
                raise ValueError('Metric {} not found!'.format(metric))

            if self.kwargs[metric] is not None:

                def batch_reward(samples, train_samples=None):
                    decoded = [mm.decode(sample, self.ord_dict)
                               for sample in samples]
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.molecules,
                                          **self.kwargs[metric])
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            else:

                def batch_reward(samples, train_samples=None):
                    decoded = [mm.decode(sample, self.ord_dict)
                               for sample in samples]
                    # print("decoded:", decoded) 
                    pct_unique = len(list(set(decoded))) / float(len(decoded))
                    rewards = reward_func(decoded, self.molecules)
                    weights = np.array([pct_unique /
                                        float(decoded.count(sample))
                                        for sample in decoded])

                    return rewards * weights

            if nbatch % 10 == 0:
                gen_samples = self.generate_samples(self.BIG_SAMPLE_NUM)
            else:
                gen_samples = self.generate_samples(self.SAMPLE_NUM)
            self.gen_loader.create_batches(gen_samples)
            results['Batch'] = nbatch
            print('\nBatch n. {}'.format(nbatch))
            print('============================\n')
            print('\nGENERATOR TRAINING')
            print('============================\n')

            # results
            mm.compute_results(batch_reward,
                               gen_samples, self.train_samples, self.ord_dict, results=results)

            for it in range(self.GEN_ITERATIONS):
                samples = self.generator.generate(self.sess)
                rewards = self.rollout.get_reward(
                    self.sess, samples, 16, self.discriminator,
                    batch_reward, self.LAMBDA)
                g_loss = self.generator.generator_step(
                    self.sess, samples, rewards)
                losses['G-loss'].append(g_loss)
                self.generator.g_count = self.generator.g_count + 1
                self.report_rewards(rewards, metric)

            self.rollout.update_params()

            # generate for discriminator
            if self.LAMBDA != 0:
                print('\nDISCRIMINATOR TRAINING')
                print('============================\n')
                for i in range(self.DIS_EPOCHS):
                    print('Discriminator epoch {}...'.format(i + 1))

                    negative_samples = self.generate_samples(self.POSITIVE_NUM)
                    dis_x_train, dis_y_train = self.dis_loader.load_train_data(
                        self.positive_samples, negative_samples)
                    dis_batches = self.dis_loader.batch_iter(
                        zip(dis_x_train, dis_y_train),
                        self.DIS_BATCH_SIZE, self.DIS_EPOCHS
                    )

                    d_losses, ce_losses, l2_losses, w_loss = [], [], [], []
                    for batch in dis_batches:
                        x_batch, y_batch = zip(*batch)
                        x_data, x_label = zip(*x_batch)
                        _, d_loss, ce_loss, l2_loss, w_loss = self.discriminator.train(
                            self.sess, x_data, y_batch, self.DIS_DROPOUT)
                        d_losses.append(d_loss)
                        ce_losses.append(ce_loss)
                        l2_losses.append(l2_loss)

                    losses['D-loss'].append(np.mean(d_losses))
                    losses['CE-loss'].append(np.mean(ce_losses))
                    losses['L2-loss'].append(np.mean(l2_losses))
                    losses['WGAN-loss'].append(np.mean(l2_losses))

                    self.discriminator.d_count = self.discriminator.d_count + 1

                print('\nDiscriminator trained.')

            results_rows.append(results)

            # save model
            if nbatch % self.EPOCH_SAVES == 0 or \
               nbatch == self.TOTAL_BATCH - 1:

                if results_rows is not None:
                    df = pd.DataFrame(results_rows)
                    df.to_csv('{}_results.csv'.format(
                        self.PREFIX), index=False)
                for key, val in losses.items():
                    v_arr = np.array(val)
                    np.save('{}_{}.npy'.format(self.PREFIX, key), v_arr)

                if nbatch is None:
                    label = 'final'
                else:
                    label = str(nbatch)

                # save models
                model_saver = tf.train.Saver()
                ckpt_dir = self.CHK_PATH

                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_file = os.path.join(
                    ckpt_dir, '{}_{}.ckpt'.format(self.PREFIX, label))
                path = model_saver.save(self.sess, ckpt_file)
                print('\nModel saved at {}'.format(path))
        
        self.ORGAN_TRAINED = True
        print('\n######### FINISHED #########')
        
    def conditional_train(self, ckpt_dir='checkpoints/', gen_steps=None):
        """Train conditional generator model
        
        Arguments
        -----------
            - ckpt_dir: Checkpoint directory
            - gen_steps: Training steps for each generator, if None use self.TOTAL_BATCH
        """
        
        # 1. First ensure model has been trained with organ_train
        if not self.ORGAN_TRAINED:
            raise ValueError('Please run organ_train first before conditional_train')

        # 2. Train classifier
        if self.CLASS_NUM > 1 and not self.PRIOR_CLASSIFIER:
            prior_classifier(self.positive_samples, ord_dict=self.ord_dict)
            print('\nClassifier training completed')
            self.PRIOR_CLASSIFIER = True
        
        if self.verbose:
            print('\nSTARTING CONDITIONAL TRAINING')
            print('============================\n')

        # Use specified steps or default value
        total_steps = gen_steps if gen_steps is not None else self.TOTAL_BATCH

        # 3. Create generator copy for each class
        generators = []
        rollouts = []
        for i in range(self.CLASS_NUM):
            if self.WGAN:
                gen = WGenerator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                               self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                               self.MAX_LENGTH, self.START_TOKEN)
            else:
                gen = Generator(self.NUM_EMB, self.GEN_BATCH_SIZE,
                              self.GEN_EMB_DIM, self.GEN_HIDDEN_DIM,
                              self.MAX_LENGTH, self.START_TOKEN)
            
            # Copy parameters from trained generator
            gen.copy_params(self.generator, self.sess)
            generators.append(gen)
            rollouts.append(ClassifyRollout(gen, 0.8, self.PAD_NUM, self.ord_dict))

        # 4. Train each class generator
        for class_idx in range(self.CLASS_NUM):
            print(f'\nTraining generator for class {class_idx}')
            
            for nbatch in tqdm(range(total_steps)):
                metric = self.EDUCATION[nbatch % self.TOTAL_BATCH]  # Loop through training program
                
                # Generate samples and calculate rewards
                gen_samples = self.generate_samples(self.SAMPLE_NUM, 
                                                 label_input=True, 
                                                 target_class=class_idx)
                # 从生成的样本中提取序列部分
                sequences = np.array([sample[0] for sample in gen_samples])  # 只使用序列部分，不包括标签
                
                # 按批次处理样本
                for batch_idx in range(0, len(sequences), self.GEN_BATCH_SIZE):
                    batch_sequences = sequences[batch_idx:batch_idx + self.GEN_BATCH_SIZE]
                    
                    # 如果最后一批不足一个批次，跳过
                    if len(batch_sequences) < self.GEN_BATCH_SIZE:
                        continue
                    
                    rewards = rollouts[class_idx].get_reward(
                        self.sess, 
                        batch_sequences,  # 传递一个批次的序列
                        16, 
                        self.discriminator,
                        self.LAMBDA
                    )
                    
                    # Update generator
                    g_loss = generators[class_idx].generator_step(
                        self.sess, batch_sequences, rewards)
                    
                    # Update rollout parameters
                    rollouts[class_idx].update_params()
                    
                    # Save model periodically
                    if nbatch % self.EPOCH_SAVES == 0 or nbatch == total_steps - 1:
                        model_saver = tf.train.Saver()
                        ckpt_file = os.path.join(
                            ckpt_dir, 
                            f'{self.PREFIX}_class{class_idx}_{nbatch}.ckpt'
                        )
                        path = model_saver.save(self.sess, ckpt_file)
                        print(f'\nClass {class_idx} generator model saved at {path}')

        # 5. Save final models
        for class_idx in range(self.CLASS_NUM):
            model_saver = tf.train.Saver()
            ckpt_file = os.path.join(
                ckpt_dir,
                f'{self.PREFIX}_class{class_idx}_final.ckpt'
            )
            path = model_saver.save(self.sess, ckpt_file)
            print(f'\nClass {class_idx} generator final model saved at {path}')
        
        print('\n######### FINISHED #########')
