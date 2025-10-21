import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class WhiteNoiseAnalysis:
    def __init__(self, fs=1000, duration=1, noise_power=1):
        """
        初始化参数
        fs: 采样频率
        duration: 信号时长(秒)
        noise_power: 噪声功率
        """
        self.fs = fs
        self.duration = duration
        self.noise_power = noise_power
        self.t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        self.N = len(self.t)
        
    def generate_white_noise(self):
        """生成高斯白噪声"""
        np.random.seed(42)  # 设置随机种子以便结果可重现
        self.white_noise = np.sqrt(self.noise_power) * np.random.randn(self.N)
        return self.white_noise
    
    def calculate_statistics(self, x):
        """计算信号的统计特性"""
        mean_val = np.mean(x)
        rms_val = np.sqrt(np.mean(x**2))
        variance_val = np.var(x)
        return mean_val, rms_val, variance_val
    
    def calculate_spectrum(self, x):
        """计算频谱"""
        spectrum = fft(x)
        freq = fftfreq(len(x), 1/self.fs)
        magnitude = np.abs(spectrum) / len(x)
        return freq, magnitude
    
    def calculate_psd(self, x, window='hann', nperseg=256):
        """计算功率谱密度"""
        f, Pxx = signal.welch(x, self.fs, window=window, nperseg=nperseg)
        return f, Pxx
    
    def calculate_autocorrelation(self, x, max_lag=None):
        """计算自相关函数"""
        if max_lag is None:
            max_lag = len(x) // 4
        autocorr = signal.correlate(x, x, mode='full')
        autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + max_lag]
        lags = np.arange(len(autocorr))
        return lags, autocorr / np.max(autocorr)
    
    def design_filters(self):
        """设计低通和带通滤波器"""
        # 低通滤波器
        nyquist = self.fs / 2
        lowpass_cutoff = 100  # Hz
        self.lowpass_b, self.lowpass_a = signal.butter(4, lowpass_cutoff/nyquist, 'low')
        
        # 带通滤波器
        bandpass_low = 150  # Hz
        bandpass_high = 250  # Hz
        self.bandpass_b, self.bandpass_a = signal.butter(4, 
                                                        [bandpass_low/nyquist, bandpass_high/nyquist], 
                                                        'bandpass')
        return self.lowpass_b, self.lowpass_a, self.bandpass_b, self.bandpass_a
    
    def apply_filters(self, x):
        """应用滤波器"""
        self.lowpass_output = signal.lfilter(self.lowpass_b, self.lowpass_a, x)
        self.bandpass_output = signal.lfilter(self.bandpass_b, self.bandpass_a, x)
        return self.lowpass_output, self.bandpass_output
    
    def plot_frequency_response(self):
        """绘制滤波器频率响应"""
        w_low, h_low = signal.freqz(self.lowpass_b, self.lowpass_a, worN=2000)
        w_band, h_band = signal.freqz(self.bandpass_b, self.bandpass_a, worN=2000)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 低通滤波器频率响应
        ax1.plot((w_low/np.pi) * (self.fs/2), 20 * np.log10(abs(h_low)))
        ax1.set_title('Lowpass Filter Frequency Response')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.grid(True)
        ax1.set_xlim(0, self.fs/2)

        # 带通滤波器频率响应
        ax2.plot((w_band/np.pi) * (self.fs/2), 20 * np.log10(abs(h_band)))
        ax2.set_title('Bandpass Filter Frequency Response')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.grid(True)
        ax2.set_xlim(0, self.fs/2)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_signal(self, x, title):
        """综合分析信号"""
        # 计算统计特性
        mean_val, rms_val, variance_val = self.calculate_statistics(x)
        
        # 计算频谱
        freq_spectrum, magnitude_spectrum = self.calculate_spectrum(x)
        
        # 计算功率谱密度
        f_psd, Pxx = self.calculate_psd(x)
        
        # 计算自相关函数
        lags, autocorr = self.calculate_autocorrelation(x)
        
        # 绘制结果
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{title} Analysis', fontsize=16)
        
        # 时域信号
        axes[0, 0].plot(self.t[:1000], x[:1000])
        axes[0, 0].set_title(f'{title} Time Domain Waveform (First 1s)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # 频谱
        positive_freq_idx = freq_spectrum >= 0
        axes[0, 1].plot(freq_spectrum[positive_freq_idx], magnitude_spectrum[positive_freq_idx])
        axes[0, 1].set_title(f'{title} Frequency Spectrum')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True)
        axes[0, 1].set_xlim(0, self.fs/2)
        
        # 功率谱密度
        axes[0, 2].plot(f_psd, Pxx)
        axes[0, 2].set_title(f'{title} Power Spectral Density')
        axes[0, 2].set_xlabel('Frequency (Hz)')
        axes[0, 2].set_ylabel('Power Spectral Density')
        axes[0, 2].grid(True)
        axes[0, 2].set_xlim(0, self.fs/2)
        
        # 自相关函数
        axes[1, 0].plot(lags/self.fs, autocorr)
        axes[1, 0].set_title(f'{title} Autocorrelation Function')
        axes[1, 0].set_xlabel('Lag (s)')
        axes[1, 0].set_ylabel('Normalized Autocorrelation')
        axes[1, 0].grid(True)
        
        # 直方图
        axes[1, 1].hist(x, bins=50, density=True, alpha=0.7)
        axes[1, 1].set_title(f'{title} Probability Density Distribution')
        axes[1, 1].set_xlabel('Amplitude')
        axes[1, 1].set_ylabel('Probability Density')
        axes[1, 1].grid(True)
        
        # 频谱对数坐标
        axes[1, 2].semilogy(freq_spectrum[positive_freq_idx], magnitude_spectrum[positive_freq_idx])
        axes[1, 2].set_title(f'{title} Frequency Spectrum (Log Scale)')
        axes[1, 2].set_xlabel('Frequency (Hz)')
        axes[1, 2].set_ylabel('Magnitude (log)')
        axes[1, 2].grid(True)
        axes[1, 2].set_xlim(0, self.fs/2)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息 - 使用中文
        print(f"\n{title} - 统计特性:")
        print(f"均值: {mean_val:.6f}")
        print(f"均方值: {rms_val:.6f}")
        print(f"方差: {variance_val:.6f}")
        
        # 打印频谱特性
        max_freq_idx = np.argmax(magnitude_spectrum[positive_freq_idx])
        max_freq = freq_spectrum[positive_freq_idx][max_freq_idx]
        print(f"频谱峰值频率: {max_freq:.2f} Hz")
        print(f"频谱峰值幅度: {magnitude_spectrum[positive_freq_idx][max_freq_idx]:.6f}")
        
        # 打印功率谱特性
        total_power = np.trapezoid(Pxx, f_psd)
        print(f"总功率 (PSD积分): {total_power:.6f}")
        print(f"PSD最大值: {np.max(Pxx):.6f}")
        
        # 打印自相关特性
        zero_lag_value = autocorr[len(autocorr)//2]
        print(f"零时延自相关值: {zero_lag_value:.6f}")
        
        return {
            'mean': mean_val, 
            'rms': rms_val, 
            'variance': variance_val,
            'freq_spectrum': freq_spectrum,
            'magnitude_spectrum': magnitude_spectrum,
            'f_psd': f_psd,
            'psd': Pxx,
            'lags': lags,
            'autocorr': autocorr
        }

def main():
    """主函数 - 执行实验分析"""
    print("=" * 60)
    print("实验3：白噪声通过线性系统分析")
    print("=" * 60)
    
    # 初始化分析器
    analyzer = WhiteNoiseAnalysis(fs=1000, duration=5, noise_power=1)
    
    # 1. 生成高斯白噪声
    print("\n1. 生成高斯白噪声...")
    white_noise = analyzer.generate_white_noise()
    
    # 2. 设计滤波器
    print("2. 设计滤波器...")
    analyzer.design_filters()
    analyzer.plot_frequency_response()
    
    # 3. 分析原始白噪声
    print("\n3. 分析原始白噪声...")
    wn_results = analyzer.analyze_signal(white_noise, "Gaussian White Noise")
    
    # 4. 应用低通滤波器并分析
    print("\n4. 分析低通滤波后的噪声...")
    lowpass_output, _ = analyzer.apply_filters(white_noise)
    lp_results = analyzer.analyze_signal(lowpass_output, "Lowpass Filtered Noise")
    
    # 5. 应用带通滤波器并分析
    print("\n5. 分析带通滤波后的噪声...")
    _, bandpass_output = analyzer.apply_filters(white_noise)
    bp_results = analyzer.analyze_signal(bandpass_output, "Bandpass Filtered Noise")
    
    # 6. 绘制功率谱密度对比图
    print("\n6. 绘制功率谱密度对比图...")
    plt.figure(figsize=(12, 8))
    plt.plot(wn_results['f_psd'], wn_results['psd'], label='Original White Noise', alpha=0.7)
    plt.plot(lp_results['f_psd'], lp_results['psd'], label='Lowpass Filtered', alpha=0.8)
    plt.plot(bp_results['f_psd'], bp_results['psd'], label='Bandpass Filtered', alpha=0.8)
    plt.title('Figure 3-1 Power Spectral Density Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, analyzer.fs/2)
    plt.show()
    
    # 7. 绘制频谱对比图
    print("\n7. 绘制频谱对比图...")
    plt.figure(figsize=(12, 8))
    positive_idx = wn_results['freq_spectrum'] >= 0
    plt.plot(wn_results['freq_spectrum'][positive_idx], wn_results['magnitude_spectrum'][positive_idx], 
             label='Original White Noise', alpha=0.7)
    plt.plot(lp_results['freq_spectrum'][positive_idx], lp_results['magnitude_spectrum'][positive_idx], 
             label='Lowpass Filtered', alpha=0.8)
    plt.plot(bp_results['freq_spectrum'][positive_idx], bp_results['magnitude_spectrum'][positive_idx], 
             label='Bandpass Filtered', alpha=0.8)
    plt.title('Figure 3-2 Frequency Spectrum Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, analyzer.fs/2)
    plt.show()
    
    # 8. 绘制自相关函数对比图
    print("\n8. 绘制自相关函数对比图...")
    plt.figure(figsize=(12, 8))
    plt.plot(wn_results['lags']/analyzer.fs, wn_results['autocorr'], label='Original White Noise', alpha=0.7)
    plt.plot(lp_results['lags']/analyzer.fs, lp_results['autocorr'], label='Lowpass Filtered', alpha=0.8)
    plt.plot(bp_results['lags']/analyzer.fs, bp_results['autocorr'], label='Bandpass Filtered', alpha=0.8)
    plt.title('Figure 3-3 Autocorrelation Function Comparison')
    plt.xlabel('Lag (s)')
    plt.ylabel('Normalized Autocorrelation')
    plt.legend()
    plt.grid(True)
    plt.xlim(-0.05, 0.05)
    plt.show()
    
    # 9. 统计特性对比表格 - 使用中文
    print("\n" + "=" * 60)
    print("统计特性对比表")
    print("=" * 60)
    print(f"{'信号类型':<15} {'均值':<12} {'均方值':<12} {'方差':<12}")
    print("-" * 60)
    print(f"{'原始白噪声':<15} {wn_results['mean']:<12.6f} {wn_results['rms']:<12.6f} {wn_results['variance']:<12.6f}")
    print(f"{'低通滤波':<15} {lp_results['mean']:<12.6f} {lp_results['rms']:<12.6f} {lp_results['variance']:<12.6f}")
    print(f"{'带通滤波':<15} {bp_results['mean']:<12.6f} {bp_results['rms']:<12.6f} {bp_results['variance']:<12.6f}")

if __name__ == "__main__":
    main()