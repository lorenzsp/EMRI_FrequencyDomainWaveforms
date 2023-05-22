import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
S_git = np.genfromtxt('./LISA_Alloc_Sh.txt')
Sh_X = CubicSpline(S_git[:,0], S_git[:,1])

try:
    import cupy as xp
except:
    import numpy as xp

# ff = 10**np.linspace(-5.0, 1.0,num=100)
# plt.figure(); plt.loglog(ff, get_sensitivity(ff)); plt.loglog(ff, Sh_X(ff),'--'); plt.show()

# redefining the LISA sensitivity
def get_sensitivity(f):
    return Sh_X(f)

def get_fft_td_windowed(signal, window, dt):
    fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(signal[0] * window )) * dt
    fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(signal[1] * window )) * dt
    return [fft_td_wave_p,fft_td_wave_c]

def get_fd_windowed(signal, window):
    transf_fd_0 = xp.fft.fftshift(xp.fft.fft(xp.fft.ifft( xp.fft.ifftshift( signal[0] ) ) * window))
    transf_fd_1 = xp.fft.fftshift(xp.fft.fft(xp.fft.ifft( xp.fft.ifftshift( signal[1] ) ) * window))
    return [transf_fd_0, transf_fd_1]


# conversion
class get_fd_waveform_fromTD():

    def __init__(self, waveform_generator, positive_frequency_mask, dt):
        self.waveform_generator = waveform_generator
        self.positive_frequency_mask = positive_frequency_mask
        self.dt = dt

    def __call__(self,*args, **kwargs):
        data_channels_td = self.waveform_generator(*args, **kwargs)
        fft_td_wave_p = xp.fft.fftshift(xp.fft.fft(data_channels_td[0]))[self.positive_frequency_mask] * self.dt
        fft_td_wave_c = xp.fft.fftshift(xp.fft.fft(data_channels_td[1]))[self.positive_frequency_mask] * self.dt
        return [fft_td_wave_p,fft_td_wave_c]

def get_colorplot(data, color_value, label):
    colors = color_value
    n_dimensions = data.shape[-1]
    # Plot the corner plot
    figure, axes = plt.subplots(n_dimensions-1, n_dimensions-1, figsize=(10, 10))

    # Custom color map
    cmap = plt.cm.get_cmap('viridis')  # Choose a color map of your preference

    for i in range(n_dimensions-1):
        for j in range(n_dimensions-1):
            if j < i:
                axes[j, i].axis('off')
            else:
                axes[j, i].scatter(data[:, i], data[:, j+1], c=colors, cmap=cmap, s=5,alpha=0.2)
                

    [axes[n_dimensions-2, i].set_xlabel(label[i]) for i in range(n_dimensions-1)]
    [axes[n_dimensions-2, i].set_yticklabels([]) for i in range(1,n_dimensions-1)]
    [axes[j, 0].set_ylabel(label[j+1]) for j in range(n_dimensions-1)]
    [axes[j, 0].set_xticklabels([]) for j in range(n_dimensions-2)]

    # Customize color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), anchor=(0.0,11.0), orientation='horizontal')
    cbar.set_label('Color', rotation=0, labelpad=15)

    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()

# usage of the colorplot
n_samples = 10000
n_dimensions = 5
data = np.random.randn(n_samples, n_dimensions)
colors = np.log(np.exp(-np.sum(data**2,axis=1)/2 ) )
label = ['var '+str(i) for i in range(n_dimensions)]
# get_colorplot(data, colors, label)

