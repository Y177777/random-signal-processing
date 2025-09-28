import numpy as np
import matplotlib.pyplot as plt

def generate_uniform(mean, var, size):
    # 均匀分布[a, b]的均值(mean)和方差(var)关系：
    # mean = (a + b) / 2, var = (b - a)^2 / 12
    """
    生成指定均值和方差的均匀分布随机数
    """

    half_range = (12 * var) ** 0.5 / 2
    a = mean - half_range
    b = mean + half_range
    data = np.random.uniform(a, b, size)
    return data, a, b

def generate_gaussian(mean, var, size):
    """
    生成指定均值和方差的高斯分布随机数
    """
    std = var ** 0.5
    data = np.random.normal(mean, std, size)
    return data

def analyze(data, name, theory_mean, theory_var):
    print(f"{name}分布:")
    print(f"  最大值: {np.max(data):.4f}")
    print(f"  最小值: {np.min(data):.4f}")
    print(f"  均值:   {np.mean(data):.4f} (理论: {theory_mean})")
    print(f"  方差:   {np.var(data):.4f} (理论: {theory_var})\n")

def main():
    mean = 5
    var = 2
    size = 10000

    # 均匀分布
    uniform_data, a, b = generate_uniform(mean, var, size)
    # 均匀分布理论均值和方差
    theory_mean_uniform = (a + b) / 2
    theory_var_uniform = ((b - a) ** 2) / 12
    analyze(uniform_data, "均匀", theory_mean_uniform, theory_var_uniform)

    # 高斯分布
    gaussian_data = generate_gaussian(mean, var, size)
    # 高斯分布理论均值和方差
    theory_mean_gaussian = mean
    theory_var_gaussian = var
    analyze(gaussian_data, "高斯", theory_mean_gaussian, theory_var_gaussian)

if __name__ == "__main__":
    main()
