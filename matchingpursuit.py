import numpy as np
import tensorflow as tf
import pickle

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5,
                            allow_growth=True)


def snr(signal, recon):
    """Returns signal-noise ratio in dB."""
    ratio = np.var(signal)/np.var(signal-recon)
    return 10*np.log10(ratio)


def dcGC(t, f):
    """Dynamic compressive gammachirp filter as defined by Irino,
    with parameters from Park as used in Charles, Kressner, & Rozell.
    The log term is regularized to log(t + 0.00001).
    t : time in seconds, greater than 0
    f : characteristic frequency in Hz
    One but not both arguments may be numpy arrays.
    """
    ERB = 0.1039*f + 24.7
    return t**3 * np.exp(-2*np.pi*1.14*ERB*t) * np.cos(2*np.pi*f*t + 0.979*np.log(t+0.000001))


class MatchingPursuer:
    """Find approximate convolutional sparse codes using matching pursuit.
    Learn kernels by gradient descent on mean squared error."""
    def __init__(self,
                 data,
                 data_dim=1,
                 nunits=32,
                 filter_time=0.05,
                 learn_rate=0.1,
                 max_iter=1000,
                 min_spike=0.1,
                 sample_rate=16000,
                 use_gammachirps=True,
                 paramfile='dummy'):
        """Parameters:
        data:       (SignalSet) object containing training data
        data_dim:   (int) Size of non-time data dimension, or 1
        nunits:     (int) number of filters/kernels to use
        filter_time:(float) length of kernels in seconds
        learn_rate: (float) gradient descent learning rate
        max_iter:   (int) maximum spikes per training example
        min_spike:  (float) minimum value of an accepted spike
        sample_rate:(float) sample rate in 1/s of data
        use_gammachirps:(float) initialize kernels to dcGCs,
                        else use 100 samples iid gaussian noise in middle
        paramfile:  (str) filename for saving parameters
        """

        self.min_spike = min_spike
        self.sample_rate = sample_rate
        self.nunits = nunits
        self.lfilter = int(filter_time * self.sample_rate)
        self.max_iter = max_iter
        self.data_dim = data_dim
        self.learn_rate = learn_rate

        self.paramfile = paramfile

        self.stims = data

        self.phi = self.initial_filters(gammachirp=use_gammachirps)
        self.graph_dict = self.build_graph()

        self.losshistory = []
        self.meanacts = np.zeros(self.nunits)
        self.L1acts = np.zeros(self.nunits)

    def initial_filters(self, gammachirp=False, seed_length=100):
        """If 1D, Return either a set of gammachirp filters or random filters,
        with seed_length iid normal samples in the middle surrounded by 0s,
        normalized. Otherwise return Gaussian noise, not normalized."""
        if self.data_dim == 1:
            if gammachirp:
                gammachirps = np.zeros([self.nunits, self.lfilter])
                freqs = np.logspace(np.log10(100), np.log10(6000), self.nunits)
                times = np.linspace(0, self.lfilter/self.sample_rate,
                                    self.lfilter)
                for ii in range(self.nunits):
                    gammachirps[ii] = dcGC(times, freqs[ii])
                filters = gammachirps
            else:
                filters = np.random.randn(self.nunits, self.lfilter)
                start = int(self.lfilter/2 - seed_length/2)
                end = start + seed_length
                filters[:, :start] = 0
                filters[:, end:] = 0
            filters /= np.sqrt(np.sum(filters**2, axis=1))[:, None]
            return filters.reshape(filters.shape+(1,))
        elif self.data_dim > 2:
            normal = tf.random_normal([self.nunits, self.lfilter, self.nfreqs])
            return normal

    def build_graph(self):
        g = tf.Graph()
        with g.as_default():
            x = tf.placeholder(tf.float32,
                               shape=[1, None, self.data_dim, 1],
                               name='input_signal')
            phi = tf.Variable(self.phi, dtype=tf.float32)

            phi_for_conv = tf.transpose(phi, [1, 2, 0])
            phi_for_conv = tf.expand_dims(phi_for_conv, 2)

            try:
                rev_phi = tf.reverse(phi, dims=[False, True, False])
            except TypeError:
                rev_phi = tf.reverse(phi, axis=[1])
            phi_for_deconv = tf.transpose(rev_phi, [1, 0, 2])

            with tf.variable_scope('inference'):
                convs = tf.nn.convolution(x,
                                          phi_for_conv,
                                          padding="VALID",
                                          name='convolutions')

            coeffs = tf.placeholder(tf.float32,
                                    shape=[1, None, self.nunits],
                                    name='coefficients')

            with tf.variable_scope('learning'):
                xhat = tf.nn.convolution(coeffs,
                                         phi_for_deconv,
                                         padding="VALID")
                resid = x - tf.expand_dims(xhat, 3)
                mse = tf.reduce_mean(tf.square(resid))
                loss = 0.5*mse
                learner = tf.train.GradientDescentOptimizer(self.learn_rate)
                learn_op = learner.minimize(loss,
                                            var_list=[phi])
                # TODO: generalize for 2D data
                normalize = phi.assign(tf.nn.l2_normalize(phi, dim=1))

            d = {'g': g,
                 'phi': phi,
                 'x': x,
                 'convs': convs,
                 'xhat': xhat,
                 'coeffs': coeffs,
                 'resid': resid,
                 'mse': mse,
                 'loss': loss,
                 'learn_op': learn_op,
                 'normalize': normalize}
            return d

    def infer(self, signal):
        d = self.graph_dict
        config = tf.ConfigProto(gpu_options=gpu_options)
        with tf.Session(graph=self.graph_dict['g'],
                        config=config) as sess:
            # some stuff here to initialize graph
            sess.run(tf.global_variables_initializer())
            sess.run(d['phi'].assign(self.phi))
            coeffs, xhat, resid, errors = self._infer(signal, sess)
        return coeffs, xhat, resid, errors

    def _infer(self, signal, sess):
        signal = signal.reshape([1, -1, self.data_dim, 1])
        resid = np.concatenate([signal,
                                np.zeros([1, self.lfilter-1,
                                          self.data_dim, 1])],
                               axis=1)
        d = self.graph_dict
        coeffs = np.zeros([1, signal.shape[1]+self.lfilter-1, self.nunits])

        errors = []

        cond = True
        iter_count = 0
        while cond:
            convs = sess.run(d['convs'], feed_dict={d['x']: resid})
            convmags = np.abs(convs)
            accept = False
            while not accept:
                winner = np.unravel_index(convmags.argmax(), convs.shape)
                coeffswinner = [winner[0], winner[1], winner[3]]
                coeffswinner[1] += self.lfilter - 1
                coeffswinner = tuple(coeffswinner)
                if coeffs[coeffswinner] != 0:
                    convmags[winner] = 0

            spike = convs[winner]
            iter_count += 1
            if np.abs(spike) < self.min_spike or iter_count > self.max_iter:
                cond = False
            if cond:
                coeffs[coeffswinner] = convs[winner]
                feed_dict = {d['x']: signal,
                             d['coeffs']: coeffs}
                resid, mse, xhat = sess.run([d['resid'], d['mse'], d['xhat']],
                                            feed_dict=feed_dict)
                errors.append(mse)
        return coeffs, xhat, resid, np.array(errors)

    def train(self, ntrials=100):
        config = tf.ConfigProto(gpu_options=gpu_options)
        d = self.graph_dict
        with tf.Session(graph=d['g'],
                        config=config) as sess:
            # some stuff here to initialize graph
            sess.run(tf.global_variables_initializer())
            sess.run(d['phi'].assign(self.phi))
            for ii in range(ntrials):
                signal = self.stims.rand_stim()
                coeffs, xhat, resid, errors = self._infer(signal, sess)
                feed_dict = {d['coeffs']: coeffs,
                             d['x']: signal.reshape([1, -1, self.data_dim, 1])}
                mse, _ = sess.run([d['mse'], d['learn_op']],
                                  feed_dict=feed_dict)
                loss = mse*signal.shape[0]/np.sum(signal**2)
                sess.run(d['normalize'])
                self.phi = sess.run(d['phi'])
                self.losshistory.append(loss)
                self.meanacts = 0.99*self.meanacts + 0.01*coeffs[0].mean(0)
                self.L1acts = 0.99*self.L1acts + 0.01*np.abs(coeffs[0]).mean(0)
                self.save()
                if ii % 50 == 0 and ii != 0:
                    print(ii)
                    self.add_noise_to_silent_units(sess, d)

    def add_noise_to_silent_units(self, sess, d):
        noise = self.initial_filters()
        for ii in range(self.nunits):
            if self.L1acts[ii] < 0.001:
                self.phi[ii] = self.phi[ii] + noise[ii]
        sess.run(d['phi'].assign(self.phi))
        sess.run(d['normalize'])
        self.phi = sess.run(d['phi'])

    def get_params(self):
        return {'max_iter': self.max_iter,
                'min_spike': self.min_spike,
                'learn_rate': self.learn_rate}

    def set_params(self, params):
        for key, val in params.items():
            setattr(self, key, val)

    def get_histories(self):
        return {'loss': self.losshistory,
                'meanacts': self.meanacts,
                'L1acts': self.L1acts}

    def set_histories(self, histories):
        self.losshistory = histories['loss']
        self.meanacts = histories['meanacts']
        self.L1acts = histories['L1acts']

    def save(self, paramfile=None):
        paramfile = paramfile or self.paramfile
        params = self.get_params()
        histories = self.get_histories()
        with open(paramfile, 'wb') as f:
            pickle.dump([self.phi, params, histories], f)

    def load(self, filename):
        self.paramfile = filename
        with open(filename, 'rb') as f:
            self.phi, params, histories = pickle.load(f)
        self.set_histories(histories)
        self.set_params(params)
