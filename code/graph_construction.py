import pandas
import numpy
import pickle
import sklearn.svm
import multiprocessing


def mean_calculation(files_in, files_out):
    """
    Calculation of mean values of regions.

    Parameters
    ----------
    files_in : list
        List of paths to the split data.
    files_out : list
        List of paths to save the mean values of regions.

    Returns
    -------
    None
    """
    for file_in, file_out in zip(files_in, files_out):
        data = pandas.read_csv(file_in, index_col=0)
        mean = data.mean(axis=1)
        mean.to_csv(file_out, index=True)


def correlation_graphs(files_in, files_out):
    """
    Creating correlation graphs.

    Parameters
    ----------
    files_in : list
        List of paths to the split data.
    files_out :list
        List of paths to save the correlation graphs.

    Returns
    -------
    None
    """
    data = pandas.read_csv(files_in[0], index_col=0)
    extract = [True if j > i else False for i in range(data.shape[0]) for j in range(data.shape[0])]

    for file_in, file_out in zip(files_in, files_out):
        data = pandas.read_csv(file_in, index_col=0)
        corr_matrix = pandas.DataFrame(numpy.corrcoef(data))
        graph_matrix = corr_matrix.stack().reset_index().iloc[extract, :]
        graph_matrix.iloc[:, 2] = (graph_matrix.iloc[:, 2] - graph_matrix.iloc[:, 2].mean()) / graph_matrix.iloc[:, 2].std()
        graph_matrix.columns = ["vertex1", "vertex2", "edge_weight"]
        graph_matrix.to_csv(file_out, index=False)


def ensemble_classifier_learning(files_in1, files_in2, files_out):
    """
    Training classifiers for computing edge weights of ensemble graphs.

    Parameters
    ----------
    files_in1 : list
        List of paths to the mean values of regions.
    files_in2 : list
        List of paths to the correlation graphs.
    files_out : list
        List of paths to save the classifiers.

    Returns
    -------
    None
    """
    def classifier_learning_(files_in1, files_in2, files_out, proc):
        data_vertex1 = pandas.read_csv(files_in1[0], index_col=0)
        data_vertex2 = pandas.read_csv(files_in1[1], index_col=0)
        data_edge1 = pandas.read_csv(files_in2[0], index_col=[0, 1])
        data_edge2 = pandas.read_csv(files_in2[1], index_col=[0, 1])
        for i in range(2, len(files_in1), 2):
            data_vertex1 = pandas.concat([data_vertex1, pandas.read_csv(files_in1[i], index_col=0)], axis=1)
            data_vertex2 = pandas.concat([data_vertex2, pandas.read_csv(files_in1[i + 1], index_col=0)], axis=1)
            data_edge1 = pandas.concat([data_edge1, pandas.read_csv(files_in2[i], index_col=[0, 1])], axis=1)
            data_edge2 = pandas.concat([data_edge2, pandas.read_csv(files_in2[i + 1], index_col=[0, 1])], axis=1)

        count = 0
        for i in range(379):
            for j in range(i + 1, 379):
                X = pandas.DataFrame({f"v{i + 1}": pandas.concat([data_vertex1.iloc[i, :], data_vertex2.iloc[i, :]], ignore_index=True),
                                      f"v{j + 1}": pandas.concat([data_vertex1.iloc[j, :], data_vertex2.iloc[j, :]], ignore_index=True),
                                      f"e{i + 1}_{j + 1}": pandas.concat([data_edge1.loc[(i, j), :], data_edge2.loc[(i, j), :]], ignore_index=True)})
                Y = [1 for k in range(int(X.shape[0] / 2))] + [2 for k in range(int(X.shape[0] / 2))]

                svc = sklearn.svm.SVC(kernel="rbf", C=1, probability=True)
                svc.fit(X, Y)
                pickle.dump(svc, open(files_out[count], 'wb'))

    files_in1_LR = [i for i in files_in1 if "LR" in i]
    files_in1_RL = [i for i in files_in1 if "RL" in i]
    files_in2_LR = [i for i in files_in2 if "LR" in i]
    files_in2_RL = [i for i in files_in2 if "RL" in i]
    file_out_LR = files_out[:71631]
    file_out_RL = files_out[71631:]

    processes = []
    processes.append(multiprocessing.Process(target=classifier_learning_, args=(files_in1_LR, files_in2_LR, file_out_LR, "LR")))
    processes.append(multiprocessing.Process(target=classifier_learning_, args=(files_in1_RL, files_in2_RL, file_out_RL, "RL")))
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def ensemble_graphs(files_in1, files_in2, files_in3, files_out):
    """
    Creating of ensemble graphs.

    Parameters
    ----------
    files_in1 : list
        List of paths to the classifiers.
    files_in2 : list
        List of paths to the mean values of regions.
    files_in3 : list
        List of paths to the correlation graphs.
    files_out : list
        List of paths to save the ensemble graphs.

    Returns
    -------

    """
    def ensemble_graphs_(files_in1, files_in2, files_in3, files_out, proc):
        data_vertex = pandas.read_csv(files_in2[0], index_col=0)
        data_edge = pandas.read_csv(files_in3[0], index_col=[0, 1])
        for i in range(1, len(files_in2)):
            data_vertex = pandas.concat([data_vertex, pandas.read_csv(files_in2[i], index_col=0)], axis=1)
            data_edge = pandas.concat([data_edge, pandas.read_csv(files_in3[i], index_col=[0, 1])], axis=1)
        vertexes = [[], []]

        count = 0
        for i in range(379):
            for j in range(i + 1, 379):
                vertexes[0].append(i + 1), vertexes[1].append(j + 1)

                svm = pickle.load(open(files_in1[count], 'rb'))
                X = pandas.DataFrame({f"v{i + 1}": data_vertex.iloc[i, :].tolist(),
                                      f"v{j + 1}": data_vertex.iloc[j, :].tolist(),
                                      f"e{i + 1}_{j + 1}": data_edge.loc[(i, j), :].tolist()})
                Y = svm.predict_proba(X)

                if i + j == 1:
                    data_edge_new = Y[:, 1] - Y[:, 0]
                else:
                    data_edge_new = numpy.vstack((data_edge_new, Y[:, 1] - Y[:, 0]))

        for count, file_out in enumerate(files_out):
            graph = pandas.DataFrame({"vertex1": vertexes[0], "vertex2": vertexes[1], "edge_weight": data_edge_new[:, count]})
            graph.to_csv(file_out, index=False)

    files_in1_LR = files_in1[:71631]
    files_in1_RL = files_in1[71631:]
    files_in2_LR_1 = files_in2[::4]
    files_in2_LR_2 = files_in2[1::4]
    files_in2_RL_1 = files_in2[2::4]
    files_in2_RL_2 = files_in2[3::4]
    files_in3_LR_1 = files_in3[::4]
    files_in3_LR_2 = files_in3[1::4]
    files_in3_RL_1 = files_in3[2::4]
    files_in3_RL_2 = files_in3[3::4]
    files_out_LR_1 = files_out[::4]
    files_out_LR_2 = files_out[1::4]
    files_out_RL_1 = files_out[2::4]
    files_out_RL_2 = files_out[3::4]

    processes = []
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_LR, files_in2_LR_1, files_in3_LR_1, files_out_LR_1, "LR 1")))
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_LR, files_in2_LR_2, files_in3_LR_2, files_out_LR_2, "LR 2")))
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_RL, files_in2_RL_1, files_in3_RL_1, files_out_RL_1, "RL 1")))
    processes.append(multiprocessing.Process(target=ensemble_graphs_,
                                             args=(files_in1_RL, files_in2_RL_2, files_in3_RL_2, files_out_RL_2, "RL 2")))
    for process in processes:
        process.start()
    for process in processes:
        process.join()
