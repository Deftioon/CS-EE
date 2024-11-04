import image
import numpy as np

img = image.read_image('data/massage1.jpg')

blocks = image.segment_image(img, 10, 10, size=(500, 500))
blocks = [image.to_grayscale(block) for block in blocks]
labels = "greetingsearth.weseetreesseasstars.wegreat.werest.weareheretosee.tolearn.toshare.seestars.greetseas."
labels = list(labels)

class dataset:
    def __init__(self, data, labels):
        self.data = self.normalize(np.array(data))
        self.labels = labels
        self.label_to_index = {label: idx for idx, label in enumerate(".oanwlshegtri")}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        self.encoded_labels = np.array(self.one_hot_encode(labels))

    def one_hot_encode(self, labels):
        one_hot_labels = []
        for label in labels:
            one_hot = [0] * len(self.label_to_index)
            one_hot[self.label_to_index[label]] = 1
            one_hot_labels.append(one_hot)
        return one_hot_labels
    
    def normalize(self, data):
        return data / 255

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.encoded_labels[index]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.data):
            result = self.data[self._index], self.encoded_labels[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

data = dataset(blocks, labels)
data.data = np.array(data.data)[:, np.newaxis, :, :]
data.encoded_labels = np.array(data.encoded_labels)[:, np.newaxis, :]

print(data.data[0])