import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import conc.profile
import helper
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    'axes.labelsize': 10,
    'xtick.labelsize':8,
    'ytick.labelsize':8
})
from scipy.optimize import curve_fit
fig_ext = '.eps'
mm = 1/25.4 # inches

def conc_profile():
    from inputs import horizon, food_surface, bact_speed, conc_horizon
    pos_x = np.linspace(0.0, horizon+bact_speed/10, num=100, endpoint=True)
    pos_xandy = lambda arr: np.vstack((arr, np.zeros(len(arr)))).T # create two columns x=arr, y=0
    env = conc.profile.DimensionTwo()
    conc_and_flag = map(env.concentration_at, pos_xandy(pos_x))
    concentration = [item[0] for item in conc_and_flag]
    fig, ax = plt.subplots(figsize=(80*mm, 60*mm))
    ax.set_title(env.grad_fun)
    ax.plot(pos_x, concentration, color='tab:blue')
    ax.set_xlabel(r'radial distance $\vert \mathbf{x}\vert $ [$\mu \mathrm{m}$]',  rotation=0, labelpad=5)
    ax.set_ylabel(r'concentration $ \, [\mathrm{\mu M}]$')
    ax.axvspan(pos_x.min(), food_surface, color='magenta')
    ax.axvspan(horizon, pos_x.max(), color='gray')
    ax.set_xlim([0,None])
    ax.set_ylim([conc_horizon,None])
    if env.grad_fun=='exp':
        plt.yscale('log')
    print(env.conc_txt)
    ax.grid(True, linestyle = ':', linewidth = 0.5)
    figpath = 'fig/conc_profile' + fig_ext
    txtpath = 'data/conc_profile.txt'
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    file = open(txtpath, "w")
    file.write(env.conc_txt)
    file.close()
    plt.show()
def write_dataframe(df, fname):
    pathcsv = 'data/' + fname + '.csv'
    pathtxt = 'data/' + fname + '.txt'
    df.to_csv(pathcsv, index=None)
    file = open(pathtxt, "w")
    text = df.to_string()
    file.write(text)
    file.close()
        
def initial_condition():
    print('plot initial_condition() '.ljust(40, '-'))
    from inputs import food_surface, horizon
    df = pd.read_csv('data/initial_condition.csv')
    x, y = df['x'].values, df['y'].values
    fig, ax = plt.subplots(figsize=(70*mm, 70*mm))
    ax.set_title("initial position [all]")
    ax.scatter(x, y, marker='o', color='tab:blue', s=0.1)
    food = plt.Circle((0,0), food_surface, color='magenta')
    horz = plt.Circle((0,0), horizon, color='gray', lw=0.5, fill=False)
    ax.add_patch(food)
    ax.add_patch(horz)
    ax.set_xlim([-1.01*horizon, 1.01*horizon])
    ax.set_ylim([-1.01*horizon, 1.01*horizon])
    ax.set_xlabel(r'$x \, [\mu \mathrm{m}]$', rotation=0, labelpad=5)
    ax.set_ylabel(r'$y \, [\mu \mathrm{m}]$', rotation=90, labelpad=5)
    ax.grid(True, linestyle = ':', linewidth = 0.5)
    ax.set_aspect('equal')
    plt.savefig('fig/init_position'+ fig_ext, bbox_inches='tight', dpi=300)
    plt.show()

def trajectory():
    print('plot trajectory() '.ljust(40, '-'))
    n_samples = 300
    traj_all = helper.unpickle_from('data/sample_trajectory')
    ytar = helper.unpickle_from('data/sample_signal_label')
    N = min(n_samples, len(traj_all))
    traj_list = traj_all[:N]
    from inputs import food_surface, horizon
    fig, ax = plt.subplots(figsize=(70*mm, 70*mm))
    ax.set_title(f'training trajectory [{N} samples]')
    for i, R in enumerate(traj_list):
        Rx = R[:,0]
        Ry = R[:,1]
        if ytar[i]==True:
            ax.plot(Rx, Ry, color='tab:green', lw=0.5)
        else:
            ax.plot(Rx, Ry, color='tomato', lw=0.5)
    food = plt.Circle((0,0), food_surface, color='magenta')
    horz = plt.Circle((0,0), horizon, color='gray', lw=0.5, fill=False)
    ax.add_patch(food)
    ax.add_patch(horz)
    ax.set_xlim([-1.01*horizon, 1.01*horizon])
    ax.set_ylim([-1.01*horizon, 1.01*horizon])
    ax.set_xlabel(r'$x \, [\mu \mathrm{m}]$', rotation=0, labelpad=5)
    ax.set_ylabel(r'$y \, [\mu \mathrm{m}]$', rotation=90, labelpad=5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle = ':', linewidth = 0.5)
    args = pd.read_csv('data/sample_args.csv', index_col=0, header=None).squeeze("columns")
    print(args)
    Dtext = r'$D_{\mathrm{rot}} =$' + str(args['D_rot']) + r'$\, [\mathrm{rad}^2/\mathrm{s}]$'
    ax.text(0.1, 0.1, Dtext, color='black', size=8, ha='left', va='bottom',
                            transform=ax.transAxes,
                            bbox=dict(facecolor='white', edgecolor='tab:blue', boxstyle='round'))
    plt.savefig('fig/trajectory'+ fig_ext , bbox_inches='tight', dpi=300)
    plt.show()

def signal():
    from inputs import T_max, memory
    dt = T_max/memory
    df_signal = pd.read_csv('data/sample_signal.csv', index_col=0)
    signal_all = df_signal.values
    nfig = 10
    signal = signal_all[0:nfig,:]
    ytar = helper.unpickle_from('data/sample_signal_label')
    time = np.arange(len(signal[0]))*dt
    args = pd.read_csv('data/sample_args.csv', index_col=0, header=None).squeeze("columns")
    Lambda, D_rot = args['Lambda'], args['D_rot']
    Dtext =  f'{D_rot}'
    title = r'$D_{\mathrm{rot}}= $' + Dtext + r'$[\mathrm{rad}^2/\mathrm{s}]$' + '$\ \ \ \ \lambda=$' +format_latex(Lambda) + r'$[1/(\mu \mathrm{M\, s})]$'
    fig, ax = plt.subplots( len(signal), figsize=(120*mm, 150*mm), sharex=True)
    ax[0].set_title(title, y=0.94)
    for i, u in enumerate(signal):
        if ytar[i]==True:
            ax[i].plot(time, u, lw=0.5, color='tab:green', label=r'+ve')
        else:
            ax[i].plot(time, u, lw=0.5, color='tomato', label=r'-ve')
        ax[i].set_yticks([0,2])
        ax[i].set_ylim([0, None])
    ax[i].set_xlabel(r'time [s]', rotation=0, labelpad=1)
    fig.subplots_adjust(hspace=0.5)
    figpath = 'fig/signal' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=200)
    plt.show()
    
def weights_for_grid():
    from inputs import Lambda_list, D_rot_list, T_max, memory
    dt = T_max/memory
    tau = np.arange(memory)*dt
    nrow, ncol = len(D_rot_list), len(Lambda_list)
    Lambda_text = [r'$\lambda = $' + format_latex(L) for L in np.array(Lambda_list)]
    Lambda_text.append(r'$\lambda \,  \rightarrow \infty$')
    color_gray = '#c6bdba'
    fig, axs = plt.subplots(nrow, ncol+1, figsize=(200*mm, 180*mm), sharex=True)
    for j in range(ncol):
        df = pd.read_csv('data/weights_mean_' + str(j) + '.csv', index_col=None)
        dfsem = pd.read_csv('data/weights_sem_' + str(j) + '.csv', index_col=None)
        for i in range(nrow):
            item = df.iloc[i]
            wsem = dfsem.iloc[i]
            b, w = item['bias'], item.loc['w0':].values
            err = wsem.loc['w0':].values
            ax = axs[i,j]
            ax.fill_between(tau, w - err/2, w + err/2, color=color_gray)
            ax.plot(tau, w, color='tab:blue', label = r'$w$', lw=0.5)
            ax.vlines(tau[-1], ymin=0, ymax=b, colors='tab:green', linestyles='solid', label=r'$b$')
            ax.axhline(y=0.0, color="black", linestyle=":", lw=0.3)
    # lambda_infinite
    df = pd.read_csv('data/weights_mean_lambda_inf' + '.csv', index_col=None)
    dfsem = pd.read_csv('data/weights_sem_lambda_inf' + '.csv', index_col=None)
    for i in range(nrow):
        ax = axs[i,-1]
        item = df.iloc[i]
        wsem = dfsem.iloc[i]
        b, w = item['bias'], item.loc['w0':].values
        err = wsem.loc['w0':].values
        #ax.fill_between(tau, w - err/2, w + err/2, color=color_gray)
        ax.plot(tau, w, color='tab:blue', label = r'$w$', lw=0.5)
        ax.vlines(tau[-1], ymin=0, ymax=b, colors='tab:green', linestyles='solid', label=r'$b$')
        ax.axhline(y=0.0, color="black", linestyle=":", lw=0.3)
    for ax in axs[-1,:]:
        ax.legend(loc='upper center', fontsize=5)
        ax.set_xlabel(r'$\tau$ [s]', rotation=0, labelpad=0)
    for i, ax in enumerate(axs[:,0]):
        Dtext = r'$D_{\mathrm{rot}} = $' + str(D_rot_list[i])
        ax.text(0.5, 0.5, Dtext, color='black', size=5, ha='center', va='center', 
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
    for j, ax in enumerate(axs[0,:]):
        ax.text(0.5, 0.8, Lambda_text[j], color='black', size=5, ha='center', va='center', 
                           transform=ax.transAxes,
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    fig.subplots_adjust(hspace=0.2, wspace=0.4)
    plt.savefig('fig/weights_grid'+ '.pdf', bbox_inches='tight', dpi=300,  pad_inches=0.1)
    plt.show()

def effective_kernel_time():
    epsilon_factor = 5/100
    from inputs import Lambda_list, T_max, memory
    dt = T_max/memory
    file_begin = 'data/weights_mean_'
    file_ending = [str(j) for j in range(len(Lambda_list))]
    file_ending.append('lambda_inf')
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df = pd.read_csv(f, index_col=None)
        args_weights = []
        for index, row in df.iterrows():
            Lambda = row['Lambda']
            D_rot = row['D_rot']
            b, w = row['bias'], row.loc['w0':].values
            wmin = np.min(w)
            wmax = np.max(w)
            epsilon = wmin*epsilon_factor
            k_wmin = np.argmin(w)
            for k_eff in range(k_wmin,memory):
                if w[k_eff]>epsilon:
                    break
            T_eff = (1+k_eff)*dt
            w_norm = w/wmax
            b_norm = b/wmax
            args_weights.append([Lambda, D_rot, T_eff] + [b_norm] + list(w_norm))
        column_names = ['Lambda', 'D_rot', 'T_eff'] + ['bias'] + [f'w{i}' for i in range(memory)]
        df_normalized = pd.DataFrame(data=args_weights, columns=column_names)
        write_dataframe(df_normalized, 'weights_normalized_' + fend)
    return

def format_latex(number):
    from decimal import Decimal
    x = Decimal(number)
    prec = 1
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    digit_first = digits[0]
    sign = '-' if tup.sign else ''
    dec = ''.join(str(i) for i in digits[1:])
    exp = x.adjusted()
    if (digit_first == 1) and (digits[1:][0] == 0):
        number_latex = f'{sign}$ 10^{exp}$'
    elif (digit_first != 1) and (digits[1:][0] == 0):
        number_latex = f'{sign}{digit_first}$\\times 10^{exp}$'
    else:
        number_latex = f'{sign}{digit_first}.{dec}$\\times 10^{exp}$'
    return(number_latex)
def weights_grouped_by_lambda_all():
    from inputs import Lambda_list, D_rot_list, T_max, memory, traj_at_fixed_r
    selected_lambda_idx = list(np.arange(len(Lambda_list))) #required for the plot.
    selected_Drot_idx = list(np.arange(len(D_rot_list))) #required for the plot.
    #
    selected_Lambda = np.array(Lambda_list)[selected_lambda_idx]
    dt = T_max/memory
    tau = np.arange(memory)*dt
    # concentration at R0, where all the trajectory begins
    env = conc.profile.DimensionTwo()
    pos = np.array([traj_at_fixed_r, 0])
    conc_and_flag = env.concentration_at(pos)
    c0 = conc_and_flag[0]
    LC0text = r'$\lambda c_0\, =\,$'
    LC0unit = r'$\,[\mathrm{s}^{-1}]$'
    #
    file_begin = 'data/weights_normalized_'
    file_ending = [str(j) for j in selected_lambda_idx]
    file_ending.append('lambda_inf')
    Lambda_text = [LC0text + format_latex(L*c0) + LC0unit for L in selected_Lambda]
    Lambda_text.append(r'$\lambda c_0 \,  \rightarrow \infty$')
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(selected_Drot_idx)))
    flag_pick = True
    nrows = len(file_ending)
    fig, axs = plt.subplots(nrows, figsize=(100*mm, nrows*30*mm), sharex=True, constrained_layout=True)
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df_all = pd.read_csv(f, index_col=None)
        df = df_all.iloc[selected_Drot_idx].reset_index(drop=False, inplace=False) # just keep a new index
        #print(df)
        ax = axs[j]
        ax.text(0.1, 0.9, Lambda_text[j], color='black', size=10, ha='left', va='top',
                                transform=ax.transAxes,
                                bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'))
        ax.axhline(y=0.0, color="black", linestyle=":", lw=0.3)
        ax.set_ylabel(r'$w(\tau)$', rotation=90, labelpad=4)
        for i, row in df.iterrows():
            D_rot = row['D_rot']
            T_eff = row['T_eff']
            Dtext = f'{D_rot}'
            w = row.loc['w0':].values
            ax.plot(tau, w, color=colors[i], label = Dtext, lw=1.0)
            ax.set_ylim([-0.4, 1.1])
            # vertical lines
            ax.axvline(x=T_eff, ymin=0.35, ymax=0.55, color=colors[i], linestyle="-", lw=1.0)
        if flag_pick:
            T_eff_pick, flag_pick = T_eff, False
    # last axis
    ax.legend(loc=(1.01, 1.5), title=r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', fontsize=8, facecolor='white', 
                  framealpha=1)
    ax.set_xlabel(r'$\tau$ [s]', size=10,  rotation=0, labelpad=2)
    # first axis
    axs[0].text(0.96, 0.92, r'memory kernels', color='white', size=10, ha='right', va='top', 
                       transform=axs[0].transAxes,
                       bbox=dict(facecolor='cadetblue', edgecolor='white', boxstyle='round'))
    # adjust the text(T_eff) with double headed arrow at a suitable position
    height = 0.27
    dist = T_eff_pick*0.45
    axs[0].annotate(r'$T_{\mathrm{eff}}$', xy=(T_eff_pick, height), xytext=(dist, height), color='tab:red',
                arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': 'tab:red'},
                va='center', size=12)
    axs[0].annotate('', xy=(dist, height), xytext=(0, height),
                arrowprops={'arrowstyle': '<-', 'lw': 0.7, 'color': 'tab:red'},
                va='center')
    figpath = 'fig/weights_grouped_all' + fig_ext
    fig.align_ylabels(axs[:])
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    plt.savefig(figpath,bbox_inches='tight', dpi=300)    
    plt.show()

def fit_loglog(D, T_eff):
    from inputs import T_max
    # T_max is the upper bound for T_eff
    # Only T_eff < T_max is valid 
    idx = np.where(T_eff<T_max)
    D, T_eff = D[idx], T_eff[idx] # remove first element
    
    X, Y = np.log10(D), np.log10(T_eff)
    popt, pcov = curve_fit(func_linear, X, Y)
    print(popt)
    return popt
def func_linear(x, slope, intercept):
    return(slope*x + intercept)

def T_eff_vs_Drot_all():
    from inputs import Lambda_list,T_max
    file_begin = 'data/weights_normalized_'
    file_ending = [str(j) for j in range(len(Lambda_list))]
    file_ending.append('lambda_inf')
    Lambda_text = [format_latex(L) for L in np.array(Lambda_list)]
    Lambda_text.append(r'$\lambda \,  \rightarrow \infty$')
    Lambdas = Lambda_list.copy()
    Lambdas.append('inf')
    colors = plt.cm.Reds(np.linspace(0.3, 1, len(Lambda_text)))
    fig, ax = plt.subplots(1, figsize=(120*mm, 80*mm))
    param = []
    print('==== fit parameters ====')
    print('    slope     intercept  ')
    print('=========================')
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df = pd.read_csv(f, index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        T_eff = df['T_eff'].values # entire column
        ax.scatter(D_rot, T_eff, color=colors[j], s=1)
        popt = fit_loglog(D_rot, T_eff)
        m, intercept = popt
        alpha = 10**intercept
        D_nonzero = D_rot[D_rot>0] # m is -ve. Division by zero-error for D^m if D<0
        T_eff_fit = alpha * (D_nonzero**m)
        ax.plot(D_nonzero, T_eff_fit, label = Ltext, color=colors[j], linewidth=0.5)
        param.append([Lambdas[j], alpha, m])
    print('=========================')
    # triangle
    color_gray = '#c6bdba'
    alpha_ = 0.25
    m_ = -1/4
    Teff_alpha_m = alpha_ * (D_nonzero**m_)
    ax.plot(D_nonzero, Teff_alpha_m, color='black', linewidth=0.3, linestyle='--')
    ind1, ind2 = 3,4
    D1, D2 = D_nonzero[ind1], D_nonzero[ind2]
    T1, T2 = Teff_alpha_m[ind1], Teff_alpha_m[ind2]
    trianglex = [D1, D2,D1, D1]
    triangley = [T2, T2, T1, T1]
    ax.fill(trianglex, triangley, color=color_gray, edgecolor='none')
    ax.text(D1, T2, r'$1$', color='black', size=8, ha='right', va='bottom')
    ax.text(D1+D2*0.2, T2, r'$4$', color='black', size=8, ha='center', va='top')
    #
    hpos = 1.8
    ax.annotate(r'', xy=(hpos, 0.3), xytext=(hpos, T_max*1), color='black',
                arrowprops={'arrowstyle': '->', 'lw': 0.3, 'color': 'black'},
                va='top', ha='center', size=10)
    ax.text(hpos*1.1, T_max*0.5, r'$\lambda$', color='black', size=8, ha='left', va='center')
    ax.set_ylabel(r'$T_{\mathrm{eff}} $ [s]', rotation=90, labelpad=-2)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
    ax.grid(True, linestyle = '--', linewidth = 0.3)
    ax.legend(loc='lower left', fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylim([None, T_max+2.0])
    ax.set_aspect('equal')
    # ax.axhline(y=T_max, color='cadetblue', linestyle="-", lw=0.5)
    ax.axhspan(T_max+0.01, 10.0, color='cadetblue', zorder=2)
    ax.text(D_nonzero[0], T_max+0.3, r'$T > T_{\mathrm{max}}$', color='white', size=10, ha='left', va='bottom')
    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/T_eff_all' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()
    df_param = pd.DataFrame(param, columns=['lambda', 'alpha', 'm'])
    helper.write_dataframe(df_param, 'fit_parameters')

#============================================================================================================
def weights_grouped_by_lambda():
     from inputs import Lambda_list, T_max, memory, traj_at_fixed_r
     selected_lambda_idx = [1,2,3] #required for the plot.
     selected_Drot_idx = [2,3,5]   #required for the plot.
     #
     selected_Lambda = np.array(Lambda_list)[selected_lambda_idx]
     dt = T_max/memory
     tau = np.arange(memory)*dt
     # concentration at R0, where all the trajectory begins
     env = conc.profile.DimensionTwo()
     pos = np.array([traj_at_fixed_r, 0])
     conc_and_flag = env.concentration_at(pos)
     c0 = conc_and_flag[0]
     LC0text = r'$\lambda c_0\, =\,$'
     LC0unit = r'$\,[\mathrm{s}^{-1}]$'
     #
     file_begin = 'data/weights_normalized_'
     file_ending = [str(j) for j in selected_lambda_idx]
     Lambda_text = []
     Lambda_text = [LC0text + format_latex(L*c0) + LC0unit for L in selected_Lambda]
     colors = plt.cm.Blues(np.linspace(0.35, 1, len(selected_Drot_idx)))
     flag_pick = True
     nrows = len(file_ending)
     fig, axs = plt.subplots(nrows, figsize=(80*mm, nrows*30*mm), sharex=True, constrained_layout=True)
     for j, fend in enumerate(file_ending):
         f = file_begin + fend + '.csv'
         df_all = pd.read_csv(f, index_col=None)
         df = df_all.iloc[selected_Drot_idx].reset_index(drop=False, inplace=False) # just keep a new index
         #print(df)
         ax = axs[j]
         ax.text(0.15, 0.9, Lambda_text[j], color='black', size=10, ha='left', va='top',
                                 transform=ax.transAxes,
                                 bbox=dict(facecolor='white', edgecolor='tab:green', boxstyle='round'))
         ax.axhline(y=0.0, color="black", linestyle=":", lw=0.3)
         ax.set_ylabel(r'$w(\tau)$', rotation=90, labelpad=4)
         for i, row in df.iterrows():
             D_rot = row['D_rot']
             T_eff = row['T_eff']
             Dtext = f'{D_rot}'
             w = row.loc['w0':].values
             ax.plot(tau, w, color=colors[i], label = Dtext, lw=1.0)
             ax.set_ylim([-0.4, 1.1])
             # vertical lines
             ax.axvline(x=T_eff, ymin=0.35, ymax=0.55, color=colors[i], linestyle="-", lw=1.0)
         if flag_pick:
             T_eff_pick, flag_pick = T_eff, False
     # last axis
     ax.legend(loc=('upper right'), title=r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', fontsize=10, facecolor='white', 
                   framealpha=1)
     ax.set_xlabel(r'$\tau$ [s]', size=10,  rotation=0, labelpad=2)
     # first axis
     # axs[0].text(0.96, 0.92, r'memory kernels', color='white', size=10, ha='right', va='top', 
     #                    transform=axs[0].transAxes,
     #                    bbox=dict(facecolor='cadetblue', edgecolor='white', boxstyle='round'))
     # adjust the text(T_eff) with double headed arrow at a suitable position
     height = 0.27
     dist = T_eff_pick*0.45
     axs[0].annotate(r'$T_{\mathrm{eff}}$', xy=(T_eff_pick, height), xytext=(dist, height), color='tab:red',
                 arrowprops={'arrowstyle': '->', 'lw': 0.7, 'color': 'tab:red'},
                 va='center', size=12)
     axs[0].annotate('', xy=(dist, height), xytext=(0, height),
                 arrowprops={'arrowstyle': '<-', 'lw': 0.7, 'color': 'tab:red'},
                 va='center')
     figpath = 'fig/weights_grouped' + fig_ext
     fig.align_ylabels(axs[:])
     fig.tight_layout(pad=0, w_pad=0, h_pad=0)
     plt.savefig(figpath,bbox_inches='tight', dpi=300)    
     plt.show()

def T_eff_vs_Drot():
    
    from inputs import Lambda_list,T_max, traj_at_fixed_r
    file_begin = 'data/weights_normalized_'
    selected_lambda_idx = [1,2,3] #required for the plot.
    selected_Lambda = np.array(Lambda_list)[selected_lambda_idx]
    file_ending = [str(j) for j in selected_lambda_idx]
    file_ending.append('lambda_inf')
    # concentration at R0, where all the trajectory begins
    env = conc.profile.DimensionTwo()
    pos = np.array([traj_at_fixed_r, 0])
    conc_and_flag = env.concentration_at(pos)
    c0 = conc_and_flag[0]
    LC0text = r'$\lambda c_0\, [\mathrm{s}^{-1}] $'
    #
    Lambda_text = [format_latex(L*c0) for L in selected_Lambda]
    Lambda_text.append(r'$ \infty$')
    colors = plt.cm.Reds(np.linspace(0.35, 1, len(Lambda_text)))
 
    fig, ax = plt.subplots(1, figsize=(80*mm, 80*mm))
    param = []
    print('==== fit parameters ====')
    print('    slope     intercept  ')
    print('=========================')
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df = pd.read_csv(f, index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        T_eff = df['T_eff'].values # entire column
        ax.scatter(D_rot, T_eff, color=colors[j], s=1)
        popt = fit_loglog(D_rot, T_eff)
        m, intercept = popt
        alpha = 10**intercept
        D_nonzero = D_rot[D_rot>0] # m is -ve. Division by zero-error for D^m if D<0
        T_eff_fit = alpha * (D_nonzero**m)
        ax.plot(D_nonzero, T_eff_fit, label = Ltext, color=colors[j], linewidth=0.5)
        param.append([Ltext, alpha, m])
    print('=========================')
    # triangle
    color_gray = '#c6bdba'
    alpha_ = 0.25
    m_ = -1/4
    Teff_alpha_m = alpha_ * (D_nonzero**m_)
    ax.plot(D_nonzero, Teff_alpha_m, color='black', linewidth=0.3, linestyle='--')
    ind1, ind2 = 3,4
    D1, D2 = D_nonzero[ind1], D_nonzero[ind2]
    T1, T2 = Teff_alpha_m[ind1], Teff_alpha_m[ind2]
    trianglex = [D1, D2,D1, D1]
    triangley = [T2, T2, T1, T1]
    ax.fill(trianglex, triangley, color=color_gray, edgecolor='none')
    ax.text(D1, T2, r'$1$', color='black', size=8, ha='right', va='bottom')
    ax.text(D1+D2*0.2, T2, r'$4$', color='black', size=8, ha='center', va='top')
    #
    hpos = 1.8
    ax.annotate(r'', xy=(hpos, 0.3), xytext=(hpos, T_max*1), color='black',
                arrowprops={'arrowstyle': '->', 'lw': 0.3, 'color': 'black'},
                va='top', ha='center', size=10)
    ax.text(hpos*1.1, T_max*0.5, r'$\lambda$', color='black', size=8, ha='left', va='center')
    ax.set_ylabel(r'$T_{\mathrm{eff}} $ [s]', rotation=90, labelpad=-4)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=0)
    # ax.grid(True, linestyle = '--', linewidth = 0.3)
    ax.minorticks_on()
    ax.legend(loc='lower left', title=LC0text, title_fontsize=5,
              fontsize=5, facecolor='white', framealpha=1)
    plt.xscale('log')
    plt.yscale('log')
    ax.set_ylim([None, T_max+0.5])
    ax.set_aspect('equal')
    ax.axhline(y=T_max, color='cadetblue', linestyle="--", lw=1.0)
    # ax.axhspan(T_max+0.01, 10.0, color='cadetblue', zorder=2)
    
    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/T_eff' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()

def score_vs_Drot():
    from inputs import Lambda_list
    file_begin = 'data/score_mean_'
    selected_lambda_idx = list(np.arange(len(Lambda_list))) #required for the plot.
    selected_Lambda = np.array(Lambda_list)[selected_lambda_idx]
    file_ending = [str(j) for j in selected_lambda_idx]
    #
    Lambda_text = [format_latex(L) for L in selected_Lambda]
    colors = plt.cm.Reds(np.linspace(0.35, 1, len(Lambda_text)))
 
    fig, ax = plt.subplots(1, figsize=(80*mm, 40*mm))
    for j, fend in enumerate(file_ending):
        f = file_begin + fend + '.csv'
        df = pd.read_csv(f, index_col=None)
        Ltext = Lambda_text[j]
        D_rot = df['D_rot'].values # entire column
        score = df['score'].values # entire column
        ax.scatter(D_rot, score*100, color=colors[j], s=1)
        ax.plot(D_rot, score*100, label = Ltext, color=colors[j], linewidth=0.5)
    hpos = 0.7*D_rot[-1]
    ax.annotate(r'', xy=(hpos, 95), xytext=(hpos, 75), color='black',
                arrowprops={'arrowstyle': '->', 'lw': 0.3, 'color': 'black'},
                va='top', ha='center', size=10)
    ax.text(hpos*1.1, 85, r'$\lambda$', color='black', size=8, ha='left', va='center')
    ax.set_ylabel(r'score [$\%$]', rotation=90, labelpad=-2)
    ax.set_xlabel(r'$D_{\mathrm{rot}}\, [\mathrm{rad}^2/s]$', rotation=0, labelpad=2)
    ax.legend(loc='lower left', title=r'$\lambda [1/(\mathrm{\mu M s})]$', title_fontsize=6,
              fontsize=6, facecolor='white', framealpha=1)
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_ylim([60, 101])
    ax.axhline(y=100, color='cadetblue', linestyle="--", lw=1.0)

    fig.subplots_adjust(hspace=0.05)
    figpath = 'fig/score' + fig_ext
    plt.savefig(figpath,bbox_inches='tight', dpi=300)
    plt.show()
    
    
if __name__ == '__main__':
    pass
                    
    
    


    
    
    

    