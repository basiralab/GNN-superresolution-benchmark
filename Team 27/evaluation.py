from sklearn.model_selection import LeaveOneOut
from dataprocessor import DataProcessor as D
from benchmark import BenchmarkUtil
import numpy as np
from os.path import exists
from matplotlib import pyplot as plt
import pandas as pd
from MatrixVectorizer import MatrixVectorizer as mv
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import community.community_louvain as community_louvain
import torch
import networkx as nx
import os
from MatrixVectorizer import MatrixVectorizer


class EvaluationUtil:
    def __init__(self, random_seed=42):
        self.model = None
        self.args = None
        self.train_method = None
        self.random_seed = random_seed
        self.calculated_metrics = {}


    def get_att_multi_dct_model(self):
        self.model, self.args = BenchmarkUtil.get_att_dct_model()
        self.train_method = BenchmarkUtil.get_att_dct_train_function()

    def vectorize_symmetric_matrix(matrix):
        """Vectorize a symmetric matrix"""
        vectorizer = MatrixVectorizer()
        return vectorizer.vectorize(matrix)

    def devectorize_symmetric_matrix(vector, size):
        """Devectorize a symmetric matrix"""
        vectorizer = MatrixVectorizer()
        return vectorizer.anti_vectorize(vector, size)

    def evaluate_model(self, model_id: str):
        """
        Evaluate the specified model with a 3-fold cross validation method,
        store the results in the dictionary self.calculated_metrics.

        :param model_id: the identifier of the model
        """

        def devectorize_symmetric_matrix(vector, size):
            """Devectorize a symmetric matrix"""
            vectorizer = MatrixVectorizer()
            return vectorizer.anti_vectorize(vector, size)
            
        def calculate_centralities(adj_matrix):
            if adj_matrix.shape[0] != adj_matrix.shape[1]:
                raise ValueError(f"Adjacency matrix is not square: shape={adj_matrix.shape}")
            print(f"Processing adjacency matrix of shape: {adj_matrix.shape}")
            adj_matrix = np.array(adj_matrix)
            G = nx.from_numpy_array(adj_matrix)
            partition = community_louvain.best_partition(G)

            # Calculate the participation coefficient with the partition
            pc_dict = participation_coefficient(G, partition)

            # Calculate averages of centrality measures
            pr = nx.pagerank(G, alpha=0.9)
            ec = nx.eigenvector_centrality_numpy(G, max_iter=100)
            bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)
            ns = np.array(list(nx.degree_centrality(G).values())) * (len(G.nodes()) - 1)
            acc = nx.average_clustering(G, weight=None)

            # Average participation coefficient
            pc_avg = np.mean(list(pc_dict.values()))

            return {
                'pr': np.mean(list(pr.values())),
                'ec': np.mean(list(ec.values())),
                'bc': np.mean(list(bc.values())),
                'ns': ns,
                'pc': pc_avg,
                'acc': acc
            }

        def participation_coefficient(G, partition):
            # Initialize dictionary for participation coefficients
            pc_dict = {}

            # Calculate participation coefficient for each node
            for node in G.nodes():
                node_degree = G.degree(node)
                if node_degree == 0:
                    pc_dict[node] = 0.0
                else:
                    # Count within-module connections
                    within_module_degree = sum(1 for neighbor in G[node] if partition[neighbor] == partition[node])
                    # Calculate participation coefficient
                    pc_dict[node] = 1 - (within_module_degree / node_degree) ** 2

            return pc_dict


        lr_train1 = pd.read_csv('../../Cluster-CV/Fold1/lr_clusterA.csv').values
        hr_train1 = pd.read_csv('../../Cluster-CV/Fold1/hr_clusterA.csv').values

        lr_train2 = pd.read_csv('../../Cluster-CV/Fold2/lr_clusterB.csv').values
        hr_train2 = pd.read_csv('../../Cluster-CV/Fold2/hr_clusterB.csv').values

        lr_train3 = pd.read_csv('../../Cluster-CV/Fold3/lr_clusterC.csv').values
        hr_train3 = pd.read_csv('../../Cluster-CV/Fold3/hr_clusterC.csv').values

        lr_train1 = torch.tensor([devectorize_symmetric_matrix(x, 160) for x in lr_train1])
        hr_train1 = torch.tensor([devectorize_symmetric_matrix(x, 268) for x in hr_train1])

        lr_train2 = torch.tensor([devectorize_symmetric_matrix(x, 160) for x in lr_train2])
        hr_train2 = torch.tensor([devectorize_symmetric_matrix(x, 268) for x in hr_train2])

        lr_train3 = torch.tensor([devectorize_symmetric_matrix(x, 160) for x in lr_train3])
        hr_train3 = torch.tensor([devectorize_symmetric_matrix(x, 268) for x in hr_train3])

# TODO:
        X = np.array([lr_train1, lr_train2, lr_train3], dtype="object")
        y = np.array([hr_train1, hr_train2, hr_train3], dtype="object")
        loo = LeaveOneOut()


        k = loo.get_n_splits(X)


        self.calculated_metrics[model_id] = np.zeros((3, 6))

        for i, (train_index, test_index) in enumerate(loo.split(X)):
            print("FOLD", i+1)
            checkFold = i + 1
            x_train = X[train_index]
            y_train = y[train_index]

            x_test = X[test_index]
            y_test = y[test_index]

            if model_id == 'AGSRNet':
                # self.get_baseline_model()
                pass
            elif model_id == 'Multi-Discriminator':
                # self.get_multi_dct_model()
                pass
            elif model_id == 'Att-Multi':
                self.get_att_multi_dct_model()

            self.train_method(self.model, np.array(x_train[0]), np.array(y_train[0]), self.args)

            y_pred_list = []
            for x in np.array(x_test[0]):
                x = torch.from_numpy(x).float()
                preds, _, _, _ = self.model(x, 160, 320)
                preds_unpad = preds[26:294, 26:294]
                y_pred_list.append(preds_unpad.detach().numpy())

            y_pred = np.array(y_pred_list)
            D.save_kaggle_csv(y_pred, f"data/predictions_fold_{i + 1}.csv")

            # So that I can test the evaluation and plotting code without models ready
            if model_id == 'evaluation_test':
                y_pred = torch.randn(y_test[0][:20].shape).numpy()
            print(y_pred.shape)
            print(y_test.shape)
            print(y_test[0].shape)

            if checkFold == 1:
                output_path='Fold1CSV-Cluster.csv'
                predicted_hr_matrices = y_pred
                true_hr_matrices = hr_train1
            elif checkFold == 2:
                output_path='Fold2CSV-Cluster.csv'
                predicted_hr_matrices = y_pred
                true_hr_matrices = hr_train2
            else:
                output_path='Fold3CSV-Cluster.csv'
                predicted_hr_matrices = y_pred
                true_hr_matrices = hr_train3

            
            num_subjects = true_hr_matrices.shape[0]
            results = []
            
            for i in range(num_subjects):
                true_matrix = true_hr_matrices[i]
                pred_matrix = predicted_hr_matrices[i]
                print("true matrix", true_matrix.shape)
                print("pred", pred_matrix.shape)

                print(f"Evaluating subject {i+1} with matrix shapes: true={true_matrix.shape}, pred={pred_matrix.shape}")

                if true_matrix.shape != pred_matrix.shape or true_matrix.shape[0] != true_matrix.shape[1]:
                    print(f"Error: Matrix shape mismatch or not square for subject {i+1}: true={true_matrix.shape}, pred={pred_matrix.shape}")
                    continue

                metrics = {
                    'ID': i + 1,
                    'MAE': mean_absolute_error(true_matrix.flatten(), pred_matrix.flatten()),
                    'PCC': pearsonr(true_matrix.flatten(), pred_matrix.flatten())[0],
                    'JSD': jensenshannon(true_matrix.flatten(), pred_matrix.flatten()),
                }

                true_metrics = calculate_centralities(true_matrix)
                pred_metrics = calculate_centralities(pred_matrix)

                for key in ['NS', 'PR', 'EC', 'BC', 'PC', 'ACC']:
                    metrics[f'MAE in {key}'] = mean_absolute_error([true_metrics[key.lower()]], [pred_metrics[key.lower()]])

                results.append(metrics)

            df = pd.DataFrame(results)
            if not df.empty:
                # Check if the file exists to decide whether to write headers
                file_exists = os.path.isfile(output_path)

                df.to_csv(output_path, mode='a', header=not file_exists, index=False)
                print(f"Results appended to {output_path}.")
            else:
                print("No data to save.")



    def flush_model(self):
        """
        Clean any possible residues, tho not likely to have.
        """
        self.model = None
        self.train_method = None
        self.args = None

    def calc_metrics(self, prediction: np.ndarray, truth: np.ndarray):
        """
        Adapting code from lib/evaluation_measure.py, calculating the
        6 metrics according to provided prediction and ground truth values.

        :param prediction: Prediction from the model
        :param truth: Ground truth output for testing data
        :return: np.ndarray(6), 6 scores of the metrics
        """
        
        prediction[prediction < 0] = 0

        # truth[truth < 0] = 0
        if (truth < 0).any():
            # Set elements less than 0 to 0
            truth = np.where(truth < 0, 0, truth)
        pred_1d_list = []
        truth_1d_list = []
        pagerank_maes = []
        eigenvector_maes = []
        betweenness_maes = []

        total = prediction.shape[0]
        for i in range(total):
            pred_graph = nx.from_numpy_array(prediction[i], edge_attr='weight')
            truth_graph = nx.from_numpy_array(truth[i], edge_attr='weight')

            pred_values = self.centrality_values(pred_graph)
            truth_values = self.centrality_values(truth_graph)

            pagerank_maes.append(mean_absolute_error(pred_values[0], truth_values[0]))
            eigenvector_maes.append(mean_absolute_error(pred_values[1], truth_values[1]))
            betweenness_maes.append(mean_absolute_error(pred_values[2], truth_values[2]))

            pred_1d_list.append(mv.vectorize(prediction[i]))
            truth_1d_list.append(mv.vectorize(truth[i]))
            print(f"Evaluating...{i}/{total}")

        pred_1d = np.concatenate(pred_1d_list)
        truth_1d = np.concatenate(truth_1d_list)

        return [
            mean_absolute_error(pred_1d, truth_1d),
            pearsonr(pred_1d, truth_1d)[0],
            jensenshannon(pred_1d, truth_1d),
            sum(pagerank_maes) / len(pagerank_maes),
            sum(eigenvector_maes) / len(eigenvector_maes),
            sum(betweenness_maes) / len(betweenness_maes)
        ]

    def save_metrics(self, file_path='data/temp_metrics'):
        """
        Save metrics dictionary to the filesystem, so no need for
        redundant evaluation when doing the plots.

        :param file_path: specify a file path to store the evaluation results.
        """
        np.savez(file_path, **self.calculated_metrics)

    def load_metrics(self, file_path='data/temp_metrics'):
        with np.load(file_path) as data:
            self.calculated_metrics = {key: data[key] for key in data.files}

    def print_metrics(self):
        """
        Print evaluation results for quick monitoring.

        """
        print(self.calculated_metrics)

    def plot_metrics(self, data=None):
        """
        Plot either given data, data stored in file, or just calculated data.

        :param data: Priority: data passed in -> data stored in file -> data in self.calculated_metrics
        """
        if data is None:
            if exists('data/temp_metrics.npz'):
                data = np.load('data/temp_metrics.npz')
            else:
                data = self.calculated_metrics

        for key, value in data.items():
            for i in range(3):
                fold_metrics = value[i]
                self.plot_single_fold(fold_metrics, i, key)

            print(value)
            mean_metrics = np.mean(value, axis=0)
            std_metrics = np.std(value, axis=0)
            self.plot_avg_fold(mean_metrics, std_metrics, key)

    def plot_metrics_final(self):
        """
        This method is used to make final plots used in the submission.
        """
        title_list = [
            'Att_Multi_Discriminator - MAE',
            'Att_Multi_Discriminator - PCC',
            'Att_Multi_Discriminator - JSD',
            'Att_Multi_Discriminator - Avg-PC',
            'Att_Multi_Discriminator - Avg-EC',
            'Att_Multi_Discriminator - Avg-BC',
        ]
        data = self.calculated_metrics.copy()
        att_multi_data = data['Att-Multi']
        print(att_multi_data)
        data_t = att_multi_data.T
        lists = [row.tolist() for row in data_t]
        for i in range(6):
            self.plot_single_metric(lists[i], title_list[i], final=True)

    def plot_metrics_test(self):
        """
        This method is used to make plots to put in the report.
        """
        title_list = [
            'Att_Multi_Discriminator - MAE',
            'Att_Multi_Discriminator - PCC',
            'Att_Multi_Discriminator - JSD',
            'Att_Multi_Discriminator - Avg-PC',
            'Att_Multi_Discriminator - Avg-EC',
            'Att_Multi_Discriminator - Avg-BC',
        ]
        data = self.calculated_metrics.copy()
        att_multi_data = data['Att-Multi']
        print(att_multi_data)
        data_t = att_multi_data.T
        lists = [row.tolist() for row in data_t]
        for i in range(6):
            self.plot_single_metric(lists[i], title_list[i])

    @staticmethod
    def plot_single_metric(data, title, final=False):
        mean_value = np.mean(data)
        std_value = np.std(data)
        x_axis = np.arange(1, 5)
        width = 0.3

        plt.bar(x_axis[0], data[0], width, label='Fold 1', color='red')
        plt.bar(x_axis[1], data[1], width, label='Fold 2', color='green')
        plt.bar(x_axis[2], data[2], width, label='Fold 3', color='blue')
        plt.bar(x_axis[3], mean_value, width, label='Mean', color='grey')

        plt.errorbar(x_axis[3], mean_value, yerr=std_value, capsize=5, color='black')

        plt.ylabel('Value')
        plt.xlabel('Items')
        plt.title(title)
        plt.xticks([1, 2, 3, 4], ['Fold 1', 'Fold 2', 'Fold 3', 'Mean/Error'])

        if final:
            plot_path = f'plots/{title}.png'
        else:
            plot_path = f'data/test_{title}.png'
        plt.savefig(plot_path, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_single_fold(data: np.ndarray, index: int, name: str):
        """
        Plot metrics for a single fold

        :param data: ndarray [6]
        :param index: index of the fold
        :param name: name of the model
        """
        x_axis = np.arange(len(data))
        x_labels = ['MAE', 'PCC', 'JSD', 'Avg-PC', 'Avg-EC', 'Avg-BC']
        plt.bar(x_axis, data)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.xticks(x_axis, x_labels)
        plt.title(f'{name} - Fold {index + 1}')
        plt.show()

    @staticmethod
    def plot_avg_fold(data: np.ndarray, std: np.ndarray, name: str):
        """
        Plot metrics for mean value across 3 folds

        :param data: ndarray[6]
        :param std: standard deviation of data
        :param name: name of the model
        """
        x_axis = np.arange(len(data))
        x_labels = ['MAE', 'PCC', 'JSD', 'Avg-PC', 'Avg-EC', 'Avg-BC']
        plt.bar(x_axis, data, yerr=std)
        plt.xlabel('Metrics')
        plt.ylabel('Value')
        plt.xticks(x_axis, x_labels)
        plt.title(f'{name} - Avg. Across Folds')
        plt.show()

    @staticmethod
    def plot_error_curve():
        """
        Plots error curve used in the report.
        """
        with np.load("data/test_baseline_training.npz") as data:
            baseline_error = data['error'][1:]
        with np.load("data/test_att_multi_training.npz") as data:
            att_error = data['error'][1:]

        x_axis = np.arange(len(baseline_error))
        plt.plot(x_axis, baseline_error, label='Baseline Error', color='grey')
        plt.plot(x_axis, att_error, label='Final Model Error', color='red')

        plt.title('Comparing Training Error between Baseline and Final Model')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()

        plt.savefig('data/test_error_curve.png', bbox_inches='tight')
        plt.show()

    @staticmethod
    def centrality_values(graph: nx.Graph):
        """
        Utility function for reusing code

        :return: tuple [pagerank, eigenvector, betweenness]
        """
        pagerank = nx.pagerank(graph, weight='weight')
        eigenvector = nx.eigenvector_centrality(graph, weight='weight')
        betweenness = nx.betweenness_centrality(graph, weight='weight')

        return list(pagerank.values()), list(eigenvector.values()), list(betweenness.values())
    