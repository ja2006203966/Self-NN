import numpy as np

class Linear:
    def __init__(self, shape, activate = None):
        # shape:(output_size, input_size)
        self.weights = ['w', 'b', 'shape', 'name']
        self.shape = shape
        self.w = np.random.normal(0, 1, shape)
        self.b = np.zeros((shape[0]))
        self.name = 'Linear'
        if activate == None:
            def fn(x):
                return x
            self.activate = fn
        else:
            self.activate = activate
    def random_decent(self, x, lr = 1e-4):
        # x: (batch_size, input_size)
        self.batch_size = x.shape[0]
        self.Lw = np.random.uniform(-1, 1, self.shape)
        self.bLw = np.tile(np.expand_dims(self.Lw, axis = 0), [self.batch_size, 1, 1])
        self.Lb = np.random.uniform(-1, 1, self.b.shape)
        self.bLb = np.tile(np.expand_dims(self.Lb, axis = 0), [self.batch_size, 1])
        w =  np.tile(np.expand_dims(self.w, axis = 0), [self.batch_size, 1, 1])
        b = np.tile(np.expand_dims(self.b, axis = 0), [self.batch_size, 1])
        x = np.expand_dims(x, axis = -1)
        return np.matmul(w - lr*self.bLw, x)[:, :, 0] + (b - lr*self.bLb)
    def update_weights(self, lr = 1e-4):
        self.w = self.w - lr*self.Lw
        self.b = self.b - lr*self.Lb
    def call(self, x):
        self.batch_size = x.shape[0]
        w =  np.tile(np.expand_dims(self.w, axis = 0), [self.batch_size, 1, 1])
        b = np.tile(np.expand_dims(self.b, axis = 0), [self.batch_size, 1])
        x = np.expand_dims(x, axis = -1)
        return np.matmul(w, x)[:, :, 0] + b

class Poly:
    def __init__(self, shape, order = [1, 2], activate = None, mean = 0, std = 1):
        # input: (batch, input_size)
        # output: (batch, input_size)
        # shape: (intput_size, node_size)
        # kernel = (node_size,)
        self.weights = ['std', 'mean', 'order', 'shape', 'name']
        self.name = 'Poly'
        def normal_dist(x, mean = 0, std = 1):
            return np.exp(-1/2*((x-mean)/std)**2)/(std*np.sqrt(2*np.pi))
        self.shape = shape
        self.input_size = shape[0]
        self.node_size = shape[-1]
        self.kernel_shape = (shape[-1],)
        # self.w = np.random.normal(0, std, (shape[-1],) )
        self.mean = np.random.normal(mean, std, (shape[-1],) )
        self.std = np.ones((shape[-1],))*std
        self.order = np.random.uniform(order[0], order[1], (shape[-1],) )
        self.norm = normal_dist
        if activate == None:
            def fn(x):
                return x
            self.activate = fn
        else:
            self.activate = activate
            
    def random_decent(self, x, lr = 1e-4):
        # x: (batch_size, input_size)
        self.batch_size = x.shape[0]
        self.Lo = np.random.uniform(-1, 1, self.kernel_shape)
        self.Lm = np.random.uniform(-1, 1, 1)
        self.Ls = np.random.uniform(-1, 1, 1)
        m = self.mean - lr*self.Lm
        s = self.std - lr*self.Ls
        o = np.tile(np.expand_dims(self.order-lr*self.Lo, axis = [0,1]), [self.batch_size, self.input_size, 1])
        x = np.expand_dims(x, axis = -1)
        x = np.tile(x, [1,1,self.node_size])
        dist = self.norm(o, mean = m, std = s)
        norm = np.expand_dims(np.sum(dist, axis = -1), axis = -1)
        return np.mean(dist/(norm)*(x**o), axis = -1)
    
    def update_weights(self, lr = 1e-4):
        self.mean = self.mean - lr*self.Lm
        self.std = self.std - lr*self.Ls
        # self.w = self.w - lr*self.Lw
        self.order = self.order - lr*self.Lo
    def call(self, x):
        self.batch_size = x.shape[0]
        o = np.tile(np.expand_dims(self.order, axis = [0,1]), [self.batch_size, self.input_size, 1])
        x = np.expand_dims(x, axis = -1)
        x = np.tile(x, [1,1,self.node_size])
        dist = self.norm(o, mean = self.mean, std = self.std)
        norm = np.expand_dims(np.sum(dist, axis = -1), axis = -1)
        return np.mean(dist/(norm)*(x**o), axis = -1) 