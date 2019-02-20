import numpy as np
dataset = np.array([[1, 400, 0.5, 1], [3, 100, 0.5, 2],
                 [2, 600, 0.54, 2], [1, 1110, 0.1, 1],
                    [11, 1110, 0.2, 3], [3, 100, 0.5, 3],
                    [13, 1000, 0.54, 3], [1, 110, 0.9, 1]
                    ])



def autoNorm(dataset):
    import numpy as np
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normDataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    normDataset = dataset - np.tile(minVals, (m, 1))
    normDataset = normDataset / np.tile(ranges, (m, 1))
    return normDataset, minVals

print(autoNorm(dataset = dataset))


