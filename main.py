import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import os


# import matplotlib as mpl


def get_objs(num, dims, types_num):
    # 设置数据集参数
    n_samples = num  # 数据点的总数
    n_features = dims  # 特征的数量，例如二维空间就是2
    centers = types_num  # 要生成的簇的数量
    cluster_std = 0.83  # 簇的标准差，决定了簇的紧密程度

    if os.path.exists("dataset_generated.npy"):
        res_load = np.load("dataset_generated.npy")
        dataset_load = res_load[:, 0:2]
        labels_load = res_load[:, 2]
        return dataset_load, labels_load

    # 生成数据集
    dataset, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std,
                                 random_state=0)

    data_to_save = np.concatenate((dataset, np.expand_dims(labels, axis=1)), axis=1)
    np.save("dataset_generated.npy", data_to_save)

    return dataset, labels


def normalization(data):  # 归一化
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range + 0.05


class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.w = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(1, output_size)

    def __call__(self, x):
        return np.transpose((np.dot(np.transpose(x, (0, 2, 1)), self.w) + self.bias), (0, 2, 1))

    def get_params(self):
        return self.w, self.bias

    def set_params(self, w: np.ndarray, bias: np.ndarray):
        self.w = w
        self.bias = bias


def show_result(answer, objs, filename=None):
    plt.scatter(objs[:, 0], objs[:, 1], c=answer)  # 显示结果
    # plt.colorbar()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def reLU(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class nn:
    def __init__(self, input_size, output_size):
        self.l1 = Linear(input_size=input_size, output_size=4)
        self.l2 = Linear(input_size=4, output_size=output_size)

    def __call__(self, x):
        tmp = self.l1(x)
        tmp = reLU(tmp)
        tmp = self.l2(tmp)
        tmp = sigmoid(tmp)
        return tmp

    def get_params(self):
        return self.l1.get_params(), self.l2.get_params()

    def set_params(self, net_params: tuple):
        self.l1.set_params(net_params[0][0], net_params[0][1])
        self.l2.set_params(net_params[1][0], net_params[1][1])


def init_dataset(dataset: np.ndarray):
    res = np.expand_dims(dataset, axis=2)
    return res


def cross(s1: tuple, s2: tuple):  # ((l1_w, l1_b), (l2_w, l2_b))
    res = (
        (
            ((s1[0][0] + s2[0][0]) / 2.0), ((s1[0][1] + s2[0][1]) / 2.0)
        ),
        (
            ((s1[1][0] + s2[1][0]) / 2.0), ((s1[1][1] + s2[1][1]) / 2.0)
        )
    )
    return res


def variation(param_package: tuple, variation_rate: float):  # 变异  ((l1_w, l1_b), (l2_w, l2_b))
    l1_w = param_package[0][0]
    l1_b = param_package[0][1]
    l2_w = param_package[1][0]
    l2_b = param_package[1][1]

    l1_r = np.random.rand()
    l2_r = np.random.rand()

    if l1_r < variation_rate:
        l1_w_scaler = np.reshape(np.ones(l1_w.shape), (-1))
        index_w = np.random.randint(0, l1_w_scaler.shape[0])
        l1_w_scaler[index_w] = (np.random.rand()* 0.3 + 1)
        l1_w_scaler = np.resize(l1_w_scaler, l1_w.shape)
        l1_w *= l1_w_scaler

        l1_b_scaler = np.reshape(np.ones(l1_b.shape), (-1))
        index_b = np.random.randint(0, l1_b_scaler.shape[0])
        l1_b_scaler[index_b] = (np.random.rand()* 0.3 + 1)
        l1_b_scaler = np.resize(l1_b_scaler, l1_b.shape)
        l1_b *= l1_b_scaler

    if l2_r < variation_rate:
        l2_w_scaler = np.reshape(np.ones(l2_w.shape), (-1))
        index_w = np.random.randint(0, l2_w_scaler.shape[0])
        l2_w_scaler[index_w] = (np.random.rand()* 0.3 + 1)
        l2_w_scaler = np.resize(l2_w_scaler, l2_w.shape)
        l2_w *= l2_w_scaler

        l2_b_scaler = np.reshape(np.ones(l2_b.shape), (-1))
        index_b = np.random.randint(0, l2_b_scaler.shape[0])
        l2_b_scaler[index_b] = (np.random.rand()* 0.3 + 1)
        l2_b_scaler = np.resize(l2_b_scaler, l2_b.shape)
        l2_b *= l2_b_scaler


def get_generation(individual_num):
    return [nn(input_size=2, output_size=1) for _ in range(individual_num)]


def test():
    net = nn(input_size=2, output_size=1)
    answer = net(dataset)
    a = 0
    net_params = net.get_params()  # 全都是地址传递哟
    test_cross = cross(net_params, net_params)
    variation(net_params, 0.05)
    a = 1
    print(net.l1.w)

    show_result(labels, dataset)  # 展示正确分类结果


def evaluate_fitness(individuals: list[nn], dataset, labels):  # 适应度计算 f = e^(-E)  E 为均方误差
    res = np.zeros((len(individuals)))
    n = len(individuals)
    N = dataset.shape[0]
    for id in range(n):
        answer = individuals[id](dataset)
        answer = answer.reshape(-1)
        fitness = np.exp((-1.0 / N) * np.sum((answer - labels.reshape(-1)) * (answer - labels.reshape(-1))))
        res[id] = fitness
    return res.copy()


if __name__ == "__main__":
    N = 100  # 数据集点数量
    individual_num = 20  # 种群个体数量
    steps_num = 999  # 迭代次数
    variation_rate = 0.08

    dataset, labels = get_objs(N, 2, 2)
    dataset = init_dataset(dataset)  # 初始化数据集

    #test()
    # 初始化种群
    generation = get_generation(individual_num)
    # 用于记录适应度
    f_list = np.zeros(steps_num)

    # 迭代之前画图
    fs_original = evaluate_fitness(generation, dataset, labels)
    best_individual_index = np.argmax(fs_original)
    answer = generation[best_individual_index](dataset).reshape(-1)
    for i in range(N):
        if answer[i] >= 0.5:
            answer[i] = 1
        else:
            answer[i] = 0

    show_result(answer, dataset)  # 显示分类结果

    for step_id in range(steps_num):
        fs = evaluate_fitness(generation, dataset, labels)  # 计算适应度

        # 自然选择
        sorted_indexes = np.argsort(fs)
        dead_fs_indexes = sorted_indexes[0:int(individual_num / 2)]
        live_fs_indexes = sorted_indexes[int(individual_num / 2)::1]

        # 把死亡的个体替换为子代
        for dead_fs_id in dead_fs_indexes:
            id_parents = np.random.choice(live_fs_indexes, 2, replace=False)

            generation[dead_fs_id].set_params(cross(generation[id_parents[0]].get_params(),
                                                    generation[id_parents[1]].get_params()))
            variation(generation[dead_fs_id].get_params(), variation_rate)  # 变异

        #  过程打印
        if step_id % 10 == 0:
            print(step_id, fs[sorted_indexes[-1]])

        f_list[step_id] = fs[sorted_indexes[-1]]

    fs_final = evaluate_fitness(generation, dataset, labels)
    best_individual_index = np.argmax(fs_final)
    answer = generation[best_individual_index](dataset).reshape(-1)

    for i in range(N):
        if answer[i] >= 0.5:
            answer[i] = 1
        else:
            answer[i] = 0

    show_result(answer, dataset)  # 显示分类结果

    # 显示适应度变化曲线
    plt.plot(f_list)
    plt.show()

    show_result(labels, dataset)  # 显示分类结果


