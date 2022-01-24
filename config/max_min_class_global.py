
class Global_Max_Min():
    def __init__(self, use_None=True, traindict=None, testdict=None):
        if use_None:
            self.max = None
            self.min = None
        else:
            self.max = traindict['x_ppl_box'].max() if traindict['y_ppl_box'].max() <= traindict['x_ppl_box'].max() else traindict['y_ppl_box'].max()
            self.min = traindict['x_ppl_box'].min() if traindict['y_ppl_box'].min() >= traindict['x_ppl_box'].min() else traindict['y_ppl_box'].min()
