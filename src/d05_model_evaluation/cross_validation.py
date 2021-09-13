import numpy as np

class CrossValidation:
    def __init__(self, event_data, folds):
        self.folds = folds
        self.event_data = event_data
        num_events = len(event_data)
        shuffled = np.random.choice(range(num_events), replace=False, size=num_events)
        folds = 5
        splits = []
        events_per_fold = int(num_events / folds)
        k = 0
        for k in range(folds-1):
            splits += [shuffled[k*events_per_fold:(k+1)*events_per_fold]]
        splits += [shuffled[(k+1)*events_per_fold:]]

        self.splits = splits

    def get_k_split(self, k):
        test_data = []
        train_data = []
        for i in range(len(self.event_data)):
            if i in self.splits[k]:
                test_data += [self.event_data[i]]
            else:
                train_data += [self.event_data[i]]

        return test_data, train_data


if __name__ == "__main__":
    from collections import OrderedDict
    from src.d04_modeling.hmm import HMM
    event_data = []
    folds = 5
    cv = CrossValidation(event_data=event_data, folds=folds)
    train_lls = OrderedDict()
    parameters = OrderedDict()
    test_lls = OrderedDict()
    for num_states in [3, 5]:
        print("Cross-validation for %i latent states..." % num_states)
        train_lls[num_states] = []
        test_lls[num_states] = []
        for k in range(folds):
            test_data, train_data = cv.get_k_split(k)
            model = HMM(num_states=num_states)
            lls_k, parameters_k = model.fit(event_data=train_data)
            # print("Last improvement: %.6f" % (lls_k[-1] - lls_k[-2]))
            train_lls[num_states] += [lls_k]
            _, test_lls_k = model.e_step(event_data=test_data, parameters=parameters_k)
            test_lls[num_states] += [test_lls_k]

        print("Training with full dataset...")
        model = HMM(num_states=num_states)
        _, parameters[num_states] = model.fit(event_data=event_data)