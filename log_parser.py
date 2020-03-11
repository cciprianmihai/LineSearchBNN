import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

rx_dict = {
    'train_loss': re.compile(r'(?<=Train Loss=)(.*)(?=,)'),
    'train_accuracy': re.compile(r'(?<=Train Accuracy=)(.*)(?=\n)'),
    'test_loss': re.compile(r'(?<=Test Loss=)(.*)(?=,)'),
    'test_accuracy': re.compile(r'(?<=Test Accuracy=)(.*)(?=\n)'),
    'net': re.compile(r'(?<=binary=\')(.*)(?=\', cuda)'),
}


def _parse_line(line):
    res = []
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            res.append([key, match])
    if res == []:
        return None
    else:
        return res


def parse_file(filepath):
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []
    net = ""
    # open the file and read through it line by line
    with open(filepath, 'r') as file_object:
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            parse = _parse_line(line)
            if parse is not None:
                for elem in parse:
                    key, val = elem
                    if key == 'train_loss':
                        train_loss.append(float(val.group(0)))
                    elif key == 'train_accuracy':
                        train_accuracy.append(float(val.group(0)) * 100)
                    elif key == 'test_loss':
                        test_loss.append(float(val.group(0)))
                    elif key == 'test_accuracy':
                        test_accuracy.append(float(val.group(0)) * 100)
                    elif key == 'net':
                        net = val.group(0)
            line = file_object.readline()

    return train_loss, train_accuracy, test_loss, test_accuracy, net


def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def make_plots(plot_name, directory='logs'):
    train_loss_plot_name = plot_name + '_train_loss.pdf'
    train_accuracy_plot_name = plot_name + '_train_accuracy.pdf'
    test_loss_plot_name = plot_name + '_test_loss.pdf'
    test_accuracy_plot_name = plot_name + '_test_accuracy.pdf'
    colors = ['red', 'blue', 'green', 'orange']
    nets = ['Adam_bnn', 'Adam_nn']
    files_list = files(directory)
    list_train_loss = []
    list_train_accuracy = []
    list_test_loss = []
    list_test_accuracy = []
    list_net = []
    for file_list in files_list:
        train_loss, train_accuracy, test_loss, test_accuracy, net = parse_file(directory + '/' + file_list)
        list_train_loss.append(train_loss)
        list_train_accuracy.append(train_accuracy)
        list_test_loss.append(test_loss)
        list_test_accuracy.append(test_accuracy)
        list_net.append(net)

    for i, list in enumerate(list_train_loss):
        df_list_train_loss = pd.DataFrame(dict(epochs=range(1, 21), loss=list))
        sns.lineplot('epochs', 'loss', data=df_list_train_loss, color=colors[i], label=list_net[i])
    plt.savefig(train_loss_plot_name)
    plt.show()

    for i, list in enumerate(list_train_accuracy):
        df_list_train_accuracy = pd.DataFrame(dict(epochs=range(1, 21), accuracies=list))
        sns.lineplot('epochs', 'accuracies', data=df_list_train_accuracy, color=colors[i], label=list_net[i])
    plt.savefig(train_accuracy_plot_name)
    plt.show()

    for i, list in enumerate(list_test_loss):
        df_list_test_loss = pd.DataFrame(dict(epochs=range(1, 21), loss=list))
        sns.lineplot('epochs', 'loss', data=df_list_test_loss, color=colors[i], label=list_net[i])
    plt.savefig(test_loss_plot_name)
    plt.show()

    for i, list in enumerate(list_test_accuracy):
        df_list_test_accuracy = pd.DataFrame(dict(epochs=range(1, 21), accuracies=list))
        sns.lineplot('epochs', 'accuracies', data=df_list_test_accuracy, color=colors[i], label=list_net[i])
    plt.savefig(test_accuracy_plot_name)
    plt.show()


make_plots('adam')
