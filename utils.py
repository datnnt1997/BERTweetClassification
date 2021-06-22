from sklearn import metrics

import os
import time
import matplotlib.pyplot as plt


def get_total_time(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def caculate_score(actuals, predicts):
    acc_score = metrics.accuracy_score(actuals, predicts)
    f1_macro_score = metrics.f1_score(actuals, predicts, average="macro")
    f1_weighted_score = metrics.f1_score(actuals, predicts, average="weighted")
    return acc_score, f1_macro_score, f1_weighted_score


def plot_loss(train_loss, eval_loss, output_dir):
    plt.plot(train_loss)
    plt.plot(eval_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Evaluate'], loc='upper left')
    plt.savefig(os.path.join(output_dir, 'train_loss.png'))
