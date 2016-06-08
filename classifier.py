import numpy as np
from copy import deepcopy


def parseData(file, M, train=True):
    """
    Return the data contained within the given quora file.

    Args:
    ----
        filename: An open file.
        M: Int repr. no. of feats.
        train: Boolean repr. training or test mode.

    Returns:
    -------
        An N x M matrix repr. train/test data &/or labels.
    """

    dims = file.readline().strip("\n").split(" ")

    N = int(dims[0])

    if train:
        M = int(dims[1])

    ids, lbls, data = [], np.zeros((N, 1), dtype='float32'), np.zeros((N, M), dtype='float32')

    # get data
    for i in xrange(N):
        line = file.readline().strip("\n").split(" ")
        ids.append(line[0])

        if not train:
            data[i] = map(lambda e: float(e.split(":")[1]), line[1:])
        else:
            lbls[i], data[i] = float(line[1]), map(lambda e: float(e.split(":")[1]), line[2:])

    if not train:
        return ids, data

    return M, ids, data, lbls


class PerceptronLayer():
    """
    A perceptron layer.
    """

    def __init__(self, no_outputs, no_inputs):
        """
        Initialize fully connected layer.

        Args:
        -----
            no_outputs: No. output classes.
            no_inputs: No. input features.
        """

        self.w, self.b = 0.01 * np.random.randn(no_outputs, no_inputs), np.zeros((no_outputs, 1))
        self.v_w, self.dw_ms, self.v_b, self.db_ms = 0, 0, 0, 0


    def bprop(self, dEdo):
        """
        Compute gradients and return sum of error from output down
        to this layer.

        Args:
        -----
            dEdo: A no_output x N array of errors from prev layers.

        Return:
        -------
            A no_inputs x N array of input errors.
        """

        N, dEds = dEdo.shape[1], dEdo * sech2(self.s)

        self.dEdw = np.dot(dEds, self.x.T) / N
        self.dEdb = np.sum(dEds, axis=1).reshape(self.b.shape) / N

        return np.dot(self.w.T, dEds)


    def update(self, eps, mu, l2, RMSProp_decay, minsq_RMSProp=0.01):
        """
        Update the weights in this layer.

        Args:
        -----
            eps: Learning rates for the weights and biases.
            mu: Momentum coefficient.
            l2: L2 regularization coefficent.
            RMSProp_decay: Decay term for the squared average.
            minsq_RMSProp: Constant added to square-root of squared average.
        """

        self.dw_ms = (RMSProp_decay * self.dw_ms) + ((1.0 - RMSProp_decay) * np.square(self.dEdw))
        self.db_ms = (RMSProp_decay * self.db_ms) + ((1.0 - RMSProp_decay) * np.square(self.dEdb))
        self.dEdw = self.dEdw / (np.sqrt(self.dw_ms) + minsq_RMSProp)
        self.dEdb = self.dEdb / (np.sqrt(self.db_ms) + minsq_RMSProp)
        self.dEdw[np.where(np.isnan(self.dEdw))] = 0
        self.dEdb[np.where(np.isnan(self.dEdb))] = 0

        self.v_w = (mu * self.v_w) - (eps * self.dEdw) - (eps * l2 * self.w)
        self.v_b = (mu * self.v_b) - (eps * self.dEdb) - (eps * l2 * self.b)
        self.w = self.w + self.v_w
        self.b = self.b + self.v_b


    def feedf(self, data):
        """
        Perform a forward pass on the input data.

        Args:
        -----
            data: An no_inputs x N array.

        Return:
        -------
            A no_outputs x N array.
        """

        self.x = data
        self.s = np.dot(self.w, self.x) + self.b

        return np.tanh(self.s)


class Mlp():
    """
    An hyperbolic tangent mlp classifier.
    """

    def __init__(self, layers):
        """
        Initialize the mlp.

        Args:
        -----
            layers: List of mlp layers.
        """
        self.layers = deepcopy(layers)


    def train(self, train, label, test, epochs, lr, mu, l2, rms, margin, display):
        """
        Train the mlp on the training set and validation set using the provided
        hyperparameters.

        Args:
        ----
            train : A no_instance x no_features matrix.
            label : A no_instance x k_class matrix.
            test : A no_instance x no_features matrix.
            epochs : Int repr. no. of training epochs.
            lr: Float repr. learning rate.
            display: Boolean repr. whether to print train progress.
        """

        prev, itrs = 0, 0

        for epoch in xrange(epochs):

            pred = self.classify(self.predict(train))
            self.backprop(pred - label)
            eps = epsilon_decay(lr, 3000, 10, itrs, 2000) #change values to suit training.
            self.update(eps, mu, l2, rms)

            err, itrs = mce(pred, label), itrs + 1

            if display:
                print '\r| Epoch: {:5d}  |  Train mce: {:.2f}  |'.format(epoch, err)
                if epoch != 0 and epoch % 100 == 0:
                    print '--------------------------------------------------------------------------------'

            # early stopping
            if epoch > 0 and np.round(err, 2) > np.round(prev, 2) + margin:
                print err, prev
                break
            else:
                prev = err

        return self.classify(self.predict(test))


    def backprop(self, dEdo):
        """
        Propagate the error gradients through the classifier.

        Args:
        -----
            dEdo: Error gradients.
        """

        error = dEdo.T
        for layer in self.layers:
            error = layer.bprop(error)


    def update(self, eps, mu, l2, rms):
        """
        Update the classifier weights using the given parameter.

        Args:
        -----
            eps: Learning rate.
        """

        for layer in self.layers:
            layer.update(eps, mu, l2, rms)


    def predict(self, data):
        """
        Return the class predictions for the given data.

        Args:
        -----
            data: An N x m array of input data.

        Returns:
        -------
            An N x 1 array of predictions.
        """

        x = data.T
        for layer in self.layers[::-1]:
            x = layer.feedf(x)

        return x.T


    def classify(self, prediction):
        """
        Peform classification using the given predictions.

        Args:
        ----
            An N x 1 array of predictions.

        Returns:
        -------
            An N x 1 array of 1s & -1s.
        """

        return np.where(prediction >= 0, 1, -1)


def sech2(data):
    """
    Find the hyperbolic secant function over the input data.

    Args:
    -----
        data : A k x N array.

    Returns:
    --------
        A k x N array.
    """
    return np.square(1 / np.cosh(data))


def mce(preds, labels):
    """
    Compute the mean classification error over the predictions.

    Args:
    ----
        preds, labels : An N x 1 array.

    Returns:
    --------
        Float repr. mean classification error.
    """

    if preds.shape != labels.shape:
        print "Mce Error: Inputs unequal."

    return 1.0 - np.average(np.where(preds == labels, 1, 0))


def normalize(data):
    """
    Normalize the given data to zero mean & unit variance.
    """

    res = (data - np.mean(data, axis=0)[np.newaxis, :])
    std = np.std(data, axis=0)[np.newaxis, :]
    std = np.where(std == 0, 1, std)
    return res / std


def epsilon_decay(eps, phi, satr, itr, intvl):
    """
    Decay the given learn rate given.

    Args:
    -----
        eps: Learning rate.
        phi: Learning decay.
        satr: Iteration to saturate learning rate or string 'Inf'.
        itr: Current iteration.
        intvl: Decay interval i.e 0 (constant), 1 (progressive) etc.

    Returns:
    --------
        The learning rate to apply.
    """
    if intvl != 0:
        i = min(itr, float(satr)) / intvl
        return eps / (1.0 + (i * phi))
    else:
        return eps


def writeAns(f, ids, ans):
    """
    Write the predictions for the test data.
    """

    for i in xrange(ans.shape[0]):
        f.write(ids[i] + " " + str(ans[i, 0]) + '\n')


def solve(filename1, filename2):
    """
    Solve the given quora challenge repr. by the given file.

    Args:
    ----
        filename: String repr. path to file.
    """

    f = open(filename1, 'r')
    m, trainIds, train, labels = parseData(f, None)
    testIds, test = parseData(f, m, False)
    f.close()

    train, test = normalize(train), normalize(test)

    mlp = Mlp([PerceptronLayer(1, m)])
    ans = mlp.train(train, labels, test, 4000, 0.0001, 0.0, 0.9, 0.0, 0.01, True)

    g = open(filename2, 'w')
    writeAns(g, testIds, ans)
