import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class GaussianSignalAnalysis:
    def __init__(self, N=10000, fs=1000, mu=2, sigma2=4):
        self.N = N
        self.fs = fs
        self.mu = mu
        self.sigma = np.sqrt(sigma2)
        self.t = np.arange(N) / fs
        self.x = None
        
    def generate_signal(self):
        np.random.seed(42)
        self.x = np.random.normal(self.mu, self.sigma, self.N)
        return self.x
    
    def analyze_statistics(self):
        if self.x is None:
            self.generate_signal()
            
        mu_est, var_est = np.mean(self.x), np.var(self.x)
        
        print("=" * 50)
        print("统计特性比较")
        print("=" * 50)
        print(f"均值:  估计={mu_est:.4f}, 理论={self.mu:.4f}, 误差={abs(mu_est-self.mu):.4f}")
        print(f"方差:  估计={var_est:.4f}, 理论={self.sigma**2:.4f}, 误差={abs(var_est-self.sigma**2):.4f}")
        
        return mu_est, var_est
    
    def plot_signal_and_pdf(self):
        if self.x is None:
            self.generate_signal()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # 时域信号
        ax1.plot(self.t[:500], self.x[:500], 'b-')
        ax1.set(xlabel='Time (s)', ylabel='Amplitude', title='Gaussian Signal')
        ax1.grid(True)
        
        # PDF比较
        ax2.hist(self.x, bins=50, density=True, alpha=0.7, label='Estimated')
        x_range = np.linspace(self.mu-4*self.sigma, self.mu+4*self.sigma, 200)
        pdf_theory = np.exp(-0.5*((x_range-self.mu)/self.sigma)**2)/(self.sigma*np.sqrt(2*np.pi))
        ax2.plot(x_range, pdf_theory, 'r-', label='Theoretical')
        ax2.set(xlabel='Amplitude', ylabel='PDF', title='PDF Comparison')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_autocorrelation(self, max_lag=100):
        if self.x is None:
            self.generate_signal()
            
        x_centered = self.x - np.mean(self.x)
        lags = np.arange(-max_lag, max_lag+1)
        
        # 自相关估计
        autocorr = np.correlate(x_centered, x_centered, mode='full') / (self.N * np.var(self.x))
        autocorr = autocorr[self.N-1-max_lag:self.N+max_lag]
        
        # 理论自相关
        autocorr_theory = (lags == 0).astype(float)
        
        plt.figure(figsize=(10, 4))
        plt.plot(lags, autocorr, 'b-', label='Estimated')
        plt.plot(lags, autocorr_theory, 'r--', label='Theoretical')
        plt.xlabel('Lag'); plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function'); plt.legend(); plt.grid(True)
        plt.show()
        
        mse = np.mean((autocorr - autocorr_theory)**2)
        print(f"自相关函数均方误差: {mse:.6f}")
    
    def plot_psd(self):
        if self.x is None:
            self.generate_signal()
            
        f, Pxx = signal.welch(self.x, self.fs, nperseg=1024)
        Pxx_theory = self.sigma**2 * np.ones_like(f)
        
        plt.figure(figsize=(10, 4))
        plt.semilogy(f, Pxx, 'b-', label='Estimated')
        plt.semilogy(f, Pxx_theory, 'r--', label='Theoretical')
        plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD')
        plt.title('Power Spectral Density'); plt.legend(); plt.grid(True)
        plt.show()
        
        mse = np.mean((Pxx[f<=self.fs/2] - Pxx_theory[f<=self.fs/2])**2)
        print(f"功率谱密度均方误差: {mse:.6f}")
    
    def run_all_analysis(self):
        print("开始高斯随机信号分析")
        print("=" * 60)
        
        self.generate_signal()
        mu_est, var_est = self.analyze_statistics()
        
        self.plot_signal_and_pdf()
        self.plot_autocorrelation()
        self.plot_psd()
        
        # 分析报告总结
        print("\n" + "=" * 60)
        print("分析报告总结")
        print("=" * 60)
        print(f"信号参数: N={self.N}, fs={self.fs}Hz, μ={self.mu}, σ²={self.sigma**2}")
        
        mu_error = abs(mu_est - self.mu)
        var_error = abs(var_est - self.sigma**2)
        mu_rel_error = mu_error / abs(self.mu) * 100
        var_rel_error = var_error / self.sigma**2 * 100
        
        print(f"均值估计误差: {mu_error:.6f}")
        print(f"方差估计误差: {var_error:.6f}")
        print(f"均值相对误差: {mu_rel_error:.2f}%")
        print(f"方差相对误差: {var_rel_error:.2f}%")
        
        if mu_rel_error < 5 and var_rel_error < 10:
            print("结果评估: ✓ 估计精度良好")
        else:
            print("结果评估: ⚠ 估计精度有待提高")

# 运行分析
if __name__ == "__main__":
    analyzer = GaussianSignalAnalysis()
    analyzer.run_all_analysis()