import numpy as np
import matplotlib.pyplot as plt
import argparse

def generate_uniform_distribution(mean, variance, size=10000):
    """
    生成指定均值和方差的均匀分布随机数
    mean = (a + b) / 2, var = (b - a)^2 / 12
    """
    a = mean - np.sqrt(3 * variance)
    b = mean + np.sqrt(3 * variance)
    
    print(f"均匀分布参数: a = {a:.4f}, b = {b:.4f}")
    print(f"理论均值: {mean}, 理论方差: {variance}")
    
    return np.random.uniform(a, b, size)

def generate_gaussian_distribution(mean, variance, size=10000):
    """
    生成指定均值和方差的高斯分布随机数
    """
    std_dev = np.sqrt(variance)
    
    print(f"高斯分布参数: 均值 = {mean}, 标准差 = {std_dev:.4f}")
    print(f"理论均值: {mean}, 理论方差: {variance}")
    
    return np.random.normal(mean, std_dev, size)

def analyze_distribution(data, distribution_name, theoretical_mean, theoretical_variance):
    """
    分析分布数据的统计特性
    """
    actual_mean = np.mean(data)
    actual_variance = np.var(data)
    actual_max = np.max(data)
    actual_min = np.min(data)
    
    print(f"\n{distribution_name}分布分析结果:")
    print("=" * 50)
    print(f"实际均值: {actual_mean:.6f}")
    print(f"理论均值: {theoretical_mean:.6f}")
    print(f"均值误差: {abs(actual_mean - theoretical_mean):.6f}")
    
    # 修复：避免除以零错误
    if theoretical_mean != 0:
        mean_relative_error = abs(actual_mean - theoretical_mean) / abs(theoretical_mean) * 100
        print(f"均值相对误差: {mean_relative_error:.4f}%")
    else:
        print(f"均值相对误差: N/A (理论均值为0)")
    
    print()
    print(f"实际方差: {actual_variance:.6f}")
    print(f"理论方差: {theoretical_variance:.6f}")
    print(f"方差误差: {abs(actual_variance - theoretical_variance):.6f}")
    
    # 修复：避免除以零错误
    if theoretical_variance != 0:
        variance_relative_error = abs(actual_variance - theoretical_variance) / theoretical_variance * 100
        print(f"方差相对误差: {variance_relative_error:.4f}%")
    else:
        print(f"方差相对误差: N/A (理论方差为0)")
    
    print()
    print(f"最大值: {actual_max:.6f}")
    print(f"最小值: {actual_min:.6f}")
    print(f"数据范围: {actual_max - actual_min:.6f}")
    print("=" * 50)
    
    return {
        'actual_mean': actual_mean,
        'actual_variance': actual_variance,
        'actual_max': actual_max,
        'actual_min': actual_min
    }

def plot_comparison(uniform_data, gaussian_data, uniform_stats, gaussian_stats, 
                   uniform_theoretical, gaussian_theoretical):
    """
    绘制理论值与实际值的比较图 - 使用英文标签
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 均匀分布直方图与理论PDF比较
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(uniform_data, bins=50, density=True, alpha=0.7, 
                               color='skyblue', edgecolor='black', label='Actual Distribution')
    
    # 绘制理论均匀分布PDF
    a = uniform_theoretical['mean'] - np.sqrt(3 * uniform_theoretical['variance'])
    b = uniform_theoretical['mean'] + np.sqrt(3 * uniform_theoretical['variance'])
    x_uniform = np.linspace(a, b, 1000)
    y_uniform = np.ones_like(x_uniform) / (b - a)  # 均匀分布PDF：1/(b-a)
    
    ax1.plot(x_uniform, y_uniform, 'r-', linewidth=2, label='Theoretical PDF')
    ax1.axvline(uniform_stats['actual_mean'], color='green', linestyle='--', 
                linewidth=2, label=f'Actual Mean: {uniform_stats["actual_mean"]:.3f}')
    ax1.axvline(uniform_theoretical['mean'], color='red', linestyle='--', 
                linewidth=2, label=f'Theoretical Mean: {uniform_theoretical["mean"]:.3f}')
    ax1.set_title('Uniform Distribution: Actual vs Theoretical')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 高斯分布直方图与理论PDF比较
    ax2 = axes[0, 1]
    n, bins, patches = ax2.hist(gaussian_data, bins=50, density=True, alpha=0.7, 
                               color='lightgreen', edgecolor='black', label='Actual Distribution')
    
    # 绘制理论高斯分布PDF
    x_gaussian = np.linspace(gaussian_stats['actual_min'], gaussian_stats['actual_max'], 1000)
    y_gaussian = (1 / (np.sqrt(2 * np.pi * gaussian_theoretical['variance'])) * 
                 np.exp(-0.5 * (x_gaussian - gaussian_theoretical['mean']) ** 2 / 
                       gaussian_theoretical['variance']))
    
    ax2.plot(x_gaussian, y_gaussian, 'r-', linewidth=2, label='Theoretical PDF')
    ax2.axvline(gaussian_stats['actual_mean'], color='green', linestyle='--', 
                linewidth=2, label=f'Actual Mean: {gaussian_stats["actual_mean"]:.3f}')
    ax2.axvline(gaussian_theoretical['mean'], color='red', linestyle='--', 
                linewidth=2, label=f'Theoretical Mean: {gaussian_theoretical["mean"]:.3f}')
    ax2.set_title('Gaussian Distribution: Actual vs Theoretical')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 均值比较条形图
    ax3 = axes[1, 0]
    distributions = ['Uniform', 'Gaussian']
    theoretical_means = [uniform_theoretical['mean'], gaussian_theoretical['mean']]
    actual_means = [uniform_stats['actual_mean'], gaussian_stats['actual_mean']]
    
    x = np.arange(len(distributions))
    width = 0.35
    
    ax3.bar(x - width/2, theoretical_means, width, label='Theoretical Mean', alpha=0.7, color='red')
    ax3.bar(x + width/2, actual_means, width, label='Actual Mean', alpha=0.7, color='green')
    
    ax3.set_xlabel('Distribution Type')
    ax3.set_ylabel('Mean')
    ax3.set_title('Mean Comparison: Theoretical vs Actual')
    ax3.set_xticks(x)
    ax3.set_xticklabels(distributions)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (theory, actual) in enumerate(zip(theoretical_means, actual_means)):
        ax3.text(i - width/2, theory + 0.1, f'{theory:.3f}', ha='center', va='bottom')
        ax3.text(i + width/2, actual + 0.1, f'{actual:.3f}', ha='center', va='bottom')
    
    # 4. 方差比较条形图
    ax4 = axes[1, 1]
    theoretical_variances = [uniform_theoretical['variance'], gaussian_theoretical['variance']]
    actual_variances = [uniform_stats['actual_variance'], gaussian_stats['actual_variance']]
    
    ax4.bar(x - width/2, theoretical_variances, width, label='Theoretical Variance', alpha=0.7, color='red')
    ax4.bar(x + width/2, actual_variances, width, label='Actual Variance', alpha=0.7, color='green')
    
    ax4.set_xlabel('Distribution Type')
    ax4.set_ylabel('Variance')
    ax4.set_title('Variance Comparison: Theoretical vs Actual')
    ax4.set_xticks(x)
    ax4.set_xticklabels(distributions)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (theory, actual) in enumerate(zip(theoretical_variances, actual_variances)):
        ax4.text(i - width/2, theory + 0.1, f'{theory:.3f}', ha='center', va='bottom')
        ax4.text(i + width/2, actual + 0.1, f'{actual:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_distributions(uniform_data, gaussian_data, uniform_mean, gaussian_mean):
    """
    绘制两种分布的直方图 - 使用英文标签
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制均匀分布
    plt.subplot(1, 2, 1)
    plt.hist(uniform_data, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Data Distribution')
    plt.axvline(uniform_mean, color='red', linestyle='--', linewidth=2, label='Theoretical Mean')
    plt.title('Uniform Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制高斯分布
    plt.subplot(1, 2, 2)
    plt.hist(gaussian_data, bins=50, alpha=0.7, color='green', edgecolor='black', label='Data Distribution')
    plt.axvline(gaussian_mean, color='red', linestyle='--', linewidth=2, label='Theoretical Mean')
    plt.title('Gaussian Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数 - 程序的入口点
    """
    # 设置默认参数
    uniform_mean = 5.0
    uniform_variance = 4.0
    gaussian_mean = 0.0
    gaussian_variance = 1.0
    sample_size = 10000
    
    # 使用argparse处理命令行参数
    parser = argparse.ArgumentParser(description='Generate and analyze uniform and Gaussian distributions')
    parser.add_argument('--uniform_mean', type=float, default=uniform_mean, 
                       help='Uniform distribution mean (default: 5.0)')
    parser.add_argument('--uniform_var', type=float, default=uniform_variance,
                       help='Uniform distribution variance (default: 4.0)')
    parser.add_argument('--gaussian_mean', type=float, default=gaussian_mean,
                       help='Gaussian distribution mean (default: 0.0)')
    parser.add_argument('--gaussian_var', type=float, default=gaussian_variance,
                       help='Gaussian distribution variance (default: 1.0)')
    parser.add_argument('--size', type=int, default=sample_size,
                       help='Sample size for each distribution (default: 10000)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Do not display plots')
    parser.add_argument('--comparison_plot', action='store_true',
                       help='Display theoretical vs actual comparison plot')
    
    args = parser.parse_args()
    
    print("随机数分布生成与分析程序")
    print("=" * 60)
    
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    
    try:
        # 生成均匀分布
        print("\n1. 均匀分布生成:")
        uniform_data = generate_uniform_distribution(args.uniform_mean, args.uniform_var, args.size)
        
        # 生成高斯分布
        print("\n2. 高斯分布生成:")
        gaussian_data = generate_gaussian_distribution(args.gaussian_mean, args.gaussian_var, args.size)
        
        # 分析均匀分布
        uniform_stats = analyze_distribution(uniform_data, "均匀", args.uniform_mean, args.uniform_var)
        
        # 分析高斯分布
        gaussian_stats = analyze_distribution(gaussian_data, "高斯", args.gaussian_mean, args.gaussian_var)
        
        # 绘制分布图
        if not args.no_plot:
            if args.comparison_plot:
                print("\n正在生成理论vs实际比较图...")
                uniform_theoretical = {'mean': args.uniform_mean, 'variance': args.uniform_var}
                gaussian_theoretical = {'mean': args.gaussian_mean, 'variance': args.gaussian_var}
                plot_comparison(uniform_data, gaussian_data, uniform_stats, gaussian_stats,
                              uniform_theoretical, gaussian_theoretical)
            else:
                print("\n正在生成分布图...")
                plot_distributions(uniform_data, gaussian_data, args.uniform_mean, args.gaussian_mean)
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        print("请检查输入的参数是否有效（方差必须为正数）")

if __name__ == "__main__":
    main()