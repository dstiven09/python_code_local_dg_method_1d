import seaborn as sns
# These settings make all plots look more beautiful
# A list of hex colours running between blue and purple
CB91_Grad_BP = [
    '#2cbdfe', '#2fb9fc', '#33b4fa', '#36b0f8', '#3aacf6', '#3da8f4',
    '#41a3f2', '#449ff0', '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
    '#568ae6', '#5986e4', '#5c81e2', '#607de0', '#6379de', '#6775dc',
    '#6a70da', '#6e6cd8', '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
    '#7f57cf', '#8353cd', '#864ecb', '#894ac9', '#8d46c7', '#9042c5',
    '#943dc3', '#9739c1', '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
    '#a924b7', '#ac20b5', '#b01bb3', '#b317b1'
]

sns.set(rc={
    'axes.axisbelow': False,
    'axes.edgecolor': 'lightgrey',
    'axes.facecolor': 'None',
    'axes.grid': False,
    'axes.labelcolor': 'dimgrey',
    'axes.spines.right': False,
    'axes.spines.top': False,
    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': 'dimgrey',
    'xtick.bottom': False,
    'xtick.color': 'dimgrey',
    'xtick.direction': 'out',
    'xtick.top': False,
    'ytick.color': 'dimgrey',
    'ytick.direction': 'out',
    'ytick.left': False,
    'ytick.right': False
},
        style='white')
sns.set_context("notebook",
                rc={
                    "font.size": 16,
                    "axes.titlesize": 20,
                    "axes.labelsize": 18
                })
