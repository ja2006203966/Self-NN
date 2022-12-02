import numpy as np
class Model:
    def __init__(self, inputs, outputs):
        self.HiddenLayers = []
        self.Inputs = inputs
        self.Outputs = outputs
        self.index = np.arange(0, inputs.shape[0], 1)
        self.loss_fn = self.MSE
        self.test_layer = -1
        self.history = {'epoch':[], 'Loss':[]}
    def add_layer(self, layer):
        self.HiddenLayers.append(layer)
    def predict(self, x):
        for i, l in enumerate(self.HiddenLayers):
            if i == self.test_layer:
                x = l.random_decent(x, lr = self.lr)
            else:
                x = l.call(x)
        return x
    def MSE(self, y_tar, y_pre):
        # y pre/tar : (batch_size, output_size)
        return np.mean((y_pre-y_tar)**2)
    def loss(self, x, y):
        y_pre = self.predict(x)
        return self.loss_fn(y, y_pre)
    def shuffle(self,):
        # index = np.arange(0, self.Inputs.shape[0], 1)
        np.random.shuffle(self.index)
        x = self.Inputs[self.index]
        N = int(len(x)/self.batch_size)
        x = x[:N*self.batch_size]
        x = np.reshape(x, (N, self.batch_size,)+x[0].shape)
        y = self.Outputs[self.index]
        y = y[:N*self.batch_size]
        y = np.reshape(y, (N, self.batch_size,)+y[0].shape)
        self.N = N
        return x, y
    def training(self, batch_size = 32, lr = 1e-4, random_times = 100):
        self.batch_size = batch_size
        self.lr = lr
        x, y = self.shuffle()
        for e in range(self.N):
        #N: number of epoches
            self.test_layer = -1
            L = self.loss(x[e], y[e])
            for i, l in enumerate(self.HiddenLayers):
                for rds in range(random_times):
                    self.test_layer = i
                    Ltest = self.loss(x[e], y[e])
                    if Ltest < L:
                        l.update_weights(lr = lr)
                        L = Ltest
            self.test_layer = -1
            self.history['epoch'].append(e)
            self.history['Loss'].append(L)
            # print("epoch{}, Loss{}".format(e, L))
    def save_weights(self, path = './model/'):
        import os
        import pickle
        if not os.path.isdir(path):
            os.mkdir(path)
        app = dict()
        app['num_of_layer'] = len(self.HiddenLayers)
        for i, l in enumerate(self.HiddenLayers):
            for wn in l.weights:
                key = 'Layer_{}_{}'.format(i, wn)
                exec("app[key] = l.{}".format(wn))
        with open(path+'Weights.plk', 'wb') as f:
            pickle.dump(app, f)
    def load_weights(self, path = './model/'):
        import pickle
        with open(path+'Weights.plk', 'rb') as f:
            app = pickle.load(f)
            for i, l in enumerate(self.HiddenLayers):
                for wn in l.weights:
                    key = 'Layer_{}_{}'.format(i, wn)
                    exec("l.{} = app[key]".format(wn))