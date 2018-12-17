import pickle as pk

from generate import predict


path_sent1 = 'feat/sent1_test.pkl'
path_label = 'feat/label_test.pkl'
with open(path_sent1, 'rb') as f:
    sent1s = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)


def test(name, sent1s, labels):
    preds = list()
    for sent1 in sent1s:
        pred = predict(sent1, name, 'search')
        preds.append(pred)
    print('\n%s:\n' % name)
    for sent1, pred, label in zip(sent1s, preds, labels):
        print('{} | {} | {}'.format(sent1, pred, label))


if __name__ == '__main__':
    test('s2s', sent1s, labels)
    test('att', sent1s, labels)
