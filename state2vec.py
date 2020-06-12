from gensim.models import word2vec
from gensim import corpora
from multiprocessing import cpu_count


SIZE = 60
WINDOW = 10
MIN_COUNT = 1
SG = 1


def convert2corpus(path):

    sequences = []
    sequence = None

    with open(path, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip('\n').split(',')
            time, origin, destination, mode = tokens[1], tokens[2], tokens[3], tokens[4]
            if int(time) == 12:
                if sequence is not None:
                    sequences.append(sequence)
                sequence = time + origin + destination + mode + ' '
            else:
                sequence += time + origin + destination + mode + ' '

    tokenized_list = [s.strip('.').split( ) for s in sequences]
    sa_dict = corpora.Dictionary()
    sa_corpus = [sa_dict.doc2bow(traj, allow_update=True) for traj in tokenized_list]

    return tokenized_list, sa_corpus


def main():

    tokenized_list, sa_corpus = convert2corpus('all_train_irl1.csv')
    print(tokenized_list)

    size_window = [(s, w) for s in [100, 200, 400] for w in range(2, 10)]

    for sw in size_window:
        model = word2vec.Word2Vec(tokenized_list, size=sw[0], window=sw[1], min_count=MIN_COUNT, sg=SG, workers=cpu_count())

        # print('parameter', sw)
        # print(model.wv.similarity('00', '01'))
        # print(model.wv.similarity('41', '43'))
        model.save('feature_model/temporal_state_action_db_'+str(sw[0])+'_'+str(sw[1])+'_'+'.model')


if __name__ == '__main__':
    main()
