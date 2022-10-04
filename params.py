
class net_param():
    def __init__(self):
        self.Height = 224
        self.Width = 224
        self.Chancel = 3
        self.Classes = 2
        self.Net_kernel_size = 3
        self.Learning_rate = 0.0001
        self.SGD_momentum = 0.0
        self.SGD_nesterov = False
        self.loss = 'SparseCategoricalCrossentropy'
        self.Train_pth = './data/train/'
        self.Vali_pth = './data/vali/'
        self.Test_pth = './test/'
        self.Model_save_pth = './model/'
        self.Batch_size = 4
        self.Epochs = 200
