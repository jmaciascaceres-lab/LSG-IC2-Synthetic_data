# IC2_LSG - Generación de datos sintéticos para validación preliminar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Semilla para reproducibilidad
RNG = np.random.default_rng(seed=2026)

# Configuración visual (IEEE-compatible, sin colores vivos)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C1 = '#002D62'   # azul USACH
C2 = '#B8312F'   # rojo acento
C3 = '#4A7C59'   # verde
C4 = '#7B5EA7'   # violeta
GRAY = '#888888'

# 1. FUNCIONES DE NORMALIZACIÓN IC2_LSG

def f1_minmax(x, xmin, xmax, invert=False):
    """F1: Min-max escalado."""
    val = np.clip((x - xmin) / (xmax - xmin), 0, 1)
    return 1 - val if invert else val

def f2_log(x, xmax):
    """F2: Escalado logarítmico."""
    x = np.clip(x, 0, None)
    return np.clip(np.log(x + 1) / np.log(xmax + 1), 0, 1)

def f3_sigmoidal(x, theta, alpha=0.15, direction=+1):
    """F3: Umbral binario con suavizado sigmoidal.

    direction (d_k) = +1 si "más es mejor" (p. ej., MVPA): valores sobre theta -> 1.
    direction (d_k) = -1 si "menos es mejor": valores sobre theta -> 0.
    """
    return np.clip(1 / (1 + np.exp(-direction * alpha * (x - theta))), 0, 1)

def f4_ordinal(x):
    """F4: Clasificación ordinal escalada."""
    x = np.asarray(x)
    out = np.where(x == 0, 0.0, np.where(x == 1, 0.5, 1.0))
    return out

def normalizar_indicadores(df):
    """Normaliza los 8 indicadores crudos a [0,1]."""
    d = df.copy()
    # Icf: condición física
    d['n_pasos']  = f2_log(d['pasos_dia'], xmax=12000)
    d['n_mvpa']   = f3_sigmoidal(d['mvpa_min_sem'], theta=150, alpha=0.04, direction=+1)  # OMS: 150 min/sem, más es mejor
    d['Icf']      = (d['n_pasos'] + d['n_mvpa']) / 2

    # Isfg: salud física general
    d['n_fc']     = f1_minmax(d['fc_reposo'], xmin=40, xmax=100, invert=True)
    d['n_sueno']  = f4_ordinal(d['calidad_sueno'])
    d['Isfg']     = (d['n_fc'] + d['n_sueno']) / 2

    # Ipma: procesos de memoria y aprendizaje
    d['n_prec_mem'] = f1_minmax(d['prec_memoria_pct'], xmin=0, xmax=100)
    d['n_vel_rec']  = f1_minmax(d['vel_recall_ms'], xmin=200, xmax=1800, invert=True)
    d['Ipma']       = (d['n_prec_mem'] + d['n_vel_rec']) / 2

    # Itd: toma de decisiones
    d['n_prec_dec'] = f1_minmax(d['prec_decision_pct'], xmin=0, xmax=100)
    d['n_t_reac']   = f1_minmax(d['tiempo_reaccion_ms'], xmin=150, xmax=1000, invert=True)
    d['Itd']        = (d['n_prec_dec'] + d['n_t_reac']) / 2

    # Índices compuestos
    d['IC_fis']  = np.sqrt(d['Icf']  * d['Isfg'])
    d['IC_ment'] = np.sqrt(d['Ipma'] * d['Itd'])
    d['IC_LSG']  = np.sqrt(d['IC_fis'] * d['IC_ment'])

    # Índice de autorregulación: media geométrica ponderada (w1 = w2 = 0.5, Tmax = 90 min)
    # tau_j = max(0, 1 - Tj/Tmax); I_AR = IC_LSG^w1 * tau_j^w2 (no compensatorio)
    W1, W2, TMAX = 0.5, 0.5, 90
    tau = np.clip(1 - d['t_juego_min'] / TMAX, 0, 1)
    d['I_AR'] = d['IC_LSG'] ** W1 * tau ** W2

    return d

# Generación de perfiles sintéticos (N=50)

N = 50

raw = pd.DataFrame({
    # Condición física:

    # Justificación: Los conteos de pasos son siempre positivos y tienen cola derecha larga
    # (la mayoría camina entre 5.000-10.000 pasos, pero algunos pueden llegar a 15.000+). El
    # log-normal captura exactamente esa asimetría.
    'pasos_dia':         np.clip(RNG.lognormal(mean=np.log(7500), sigma=0.45, size=N), 500, 15000),

    # Justificación: La actividad física moderada-vigorosa (MVPA) también es positiva y asimétrica.
    # La gamma con shape=2.5 produce una cola más larga que la normal, realista para la población
    # que mezcla sedentarios y activos. Media teórica: alpha * beta = 2.5 * 55 = 137.5 min/sem.
    'mvpa_min_sem':      np.clip(RNG.gamma(shape=2.5, scale=55, size=N), 0, 420),
    
    # --------

    # Salud física general:

    # Justificación: La FC en reposo en adultos jóvenes sanos sigue una distribución aprox. simétrica.
    # Rango típico entre 55-85 lpm, con media en 68 aprox. Determinamos un rango entre [40, 110] para
    # evitar valores fisiológicamente imposibles.
    'fc_reposo':         np.clip(RNG.normal(loc=68, scale=9, size=N), 40, 110),

    # Justificación: La calidad de sueño percibida es ordinal, no continua. Las probabilidades reflejan
    # que en la población universitaria-joven hay un porcentaje significativo con sueño regular (categoria 1).
    'calidad_sueno':     RNG.choice([0, 1, 2], size=N, p=[0.30, 0.40, 0.30]),

    # --------

    # Memoria y aprendizaje:

    # Justificación: En adultos jóvenes, el rendimiento en pruebas cognitivas tiende a ser alto y con
    # techo (sesgo hacia valores altos). La beta(6,2) tiene media teórica 6/(6+2)=0.75, representando
    # un buen nivel de rendimiento con variabilidad moderada.
    'prec_memoria_pct':  np.clip(RNG.beta(a=6, b=2, size=N) * 100, 0, 100),

    # Justificación: Los tiempos de respuesta en pruebas de memoria siguen distribuciones aprox. normales
    # en el centro del rango. El rango elimina tiempos imposiblemente rápidos o lentos.
    'vel_recall_ms':     np.clip(RNG.normal(loc=850, scale=200, size=N), 200, 1800),

    # --------

    # Toma de decisiones:

    'prec_decision_pct': np.clip(RNG.beta(a=5, b=2, size=N) * 100, 0, 100),

    # Justificación: Los tiempos de reacción tienen una cola derecha característica (reacciones lentas
    # ocasionales), que la normal no captura bien. La suma de una gaussiana con una exponencial, llamada
    # distribución ex-gaussiana, es el modelo estándar en psicología cognitiva para tiempos de reacción.
    # El componente exponencial (scale=90), modela esos "outliers lentos" realistas.
    'tiempo_reaccion_ms':np.clip(RNG.normal(loc=450, scale=85, size=N) +
                                 RNG.exponential(scale=90, size=N), 150, 1000),
    # Tiempo de juego (para I_AR)
    't_juego_min':       np.clip(RNG.normal(loc=65, scale=30, size=N), 10, 180),
})

df = normalizar_indicadores(raw)

# 3. ARQUETIPOS (5 perfiles representativos)

arquetipos_raw = pd.DataFrame({
    'arquetipo':          ['Balanced', 'Physical↑ Mental↓', 'Physical↓ Mental↑',
                           'Low general', 'Optimal'],
    'pasos_dia':          [8200,  11500,  2800,  2200, 11000],
    'mvpa_min_sem':       [185,   310,    40,    25,   300],
    'fc_reposo':          [65,    58,     75,    85,   55],
    'calidad_sueno':      [2,     2,      1,     0,    2],
    'prec_memoria_pct':   [72,    38,     88,    34,   92],
    'vel_recall_ms':      [820,   1100,   550,   1400, 480],
    'prec_decision_pct':  [68,    32,     85,    30,   91],
    'tiempo_reaccion_ms': [440,   680,    350,   780,  300],
    't_juego_min':        [60,    70,     55,    90,   45],
})

arq = normalizar_indicadores(arquetipos_raw.drop('arquetipo', axis=1))
arq['arquetipo'] = arquetipos_raw['arquetipo'].values

# Imprimir tabla de arquetipos
print("\n" + "="*70)
print("TABLA DE ARQUETIPOS - IC2_LSG")
print("="*70)
cols_show = ['arquetipo','Icf','Isfg','Ipma','Itd','IC_fis','IC_ment','IC_LSG','I_AR']
print(arq[cols_show].round(3).to_string(index=False))

# Reglas activadas por arquetipo
def evaluar_reglas(row):
    activas = []
    if row['IC_fis'] >= 0.60: activas.append('R1')
    if row['IC_ment'] >= 0.55: activas.append('R2')
    if row['IC_LSG'] >= 0.60: activas.append('R3')
    if row['IC_fis'] < 0.40:  activas.append('R4')
    if row['IC_ment'] < 0.35: activas.append('R5')
    if row['IC_fis'] >= 0.70 and row['IC_ment'] >= 0.65: activas.append('R6')
    return ', '.join(activas) if activas else '-'

arq['reglas'] = arq.apply(evaluar_reglas, axis=1)
print("\nReglas activadas por arquetipo:")
print(arq[['arquetipo','IC_fis','IC_ment','IC_LSG','reglas']].round(3).to_string(index=False))

# Estadísticas del conjunto de N=50
print("\n" + "="*70)
print(f"ESTADÍSTICAS DESCRIPTIVAS - N={N} perfiles sintéticos")
print("="*70)
desc_cols = ['IC_fis','IC_ment','IC_LSG','I_AR']
print(df[desc_cols].describe().round(3))

# 4. FIGURAS PARA EL PAPER

# Fig 2: Distribución de índices en N=50
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
labels = ['$IC_{physical}$', '$IC_{mental}$', '$IC_{LSG}$']
cols   = ['IC_fis', 'IC_ment', 'IC_LSG']
colors = [C1, C3, C2]

for ax, col, lbl, col_c in zip(axes, cols, labels, colors):
    data = df[col]
    ax.hist(data, bins=12, color=col_c, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axvline(data.mean(), color='black', linestyle='--', linewidth=1.0,
               label=f'$\\mu={data.mean():.2f}$')
    ax.axvline(data.median(), color=GRAY, linestyle=':', linewidth=1.0,
               label=f'$\\tilde{{x}}={data.median():.2f}$')
    ax.set_xlabel(lbl)
    ax.set_ylabel('Frequency' if col == 'IC_fis' else '')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=6.5)
    ax.set_title(lbl, pad=3)

fig.suptitle(f'Distribution of $IC_{{LSG}}$ indices ($N={N}$ synthetic profiles)',
             fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig('figures/fig_distribucion.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('figures/fig_distribucion.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("\n[OK] fig_distribucion guardada")

# Fig 3: Geométrico vs Aritmético - desequilibrio
IC_fis_vals = np.linspace(0.05, 0.95, 200)
IC_ment_fixed = 0.20   # mental bajo fijo

geom  = np.sqrt(IC_fis_vals * IC_ment_fixed)
aritm = (IC_fis_vals + IC_ment_fixed) / 2

fig, ax = plt.subplots(figsize=(3.2, 2.6))
ax.plot(IC_fis_vals, geom,  color=C2, linewidth=1.8,
        label='Geometric mean $IC_{LSG}$')
ax.plot(IC_fis_vals, aritm, color=GRAY, linewidth=1.4,
        linestyle='--', label='Arithmetic mean')
ax.axhline(IC_ment_fixed, color=C4, linewidth=0.9, linestyle=':',
           label=f'$IC_{{ment}}={IC_ment_fixed}$ (fixed)')
ax.fill_between(IC_fis_vals, geom, aritm, alpha=0.12, color=C2,
                label='Imbalance penalty')
ax.set_xlabel('$IC_{physical}$', fontsize=11)
ax.set_ylabel('Global index', fontsize=11)
ax.set_xlim(0, 1); ax.set_ylim(0, 0.75)
ax.legend(frameon=False, fontsize=6.5, loc='upper left')
ax.set_title('Penalty for imbalance', fontsize=8.5)
fig.tight_layout()
fig.savefig('figures/fig_geom_vs_aritm.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('figures/fig_geom_vs_aritm.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("[OK] fig_geom_vs_aritm guardada\n")

# Fig 4: Sensibilidad - un indicador varía, resto fijo
baseline = {
    'pasos_dia': 7500, 'mvpa_min_sem': 150,
    'fc_reposo': 68,   'calidad_sueno': 1,
    'prec_memoria_pct': 72, 'vel_recall_ms': 820,
    'prec_decision_pct': 68, 'tiempo_reaccion_ms': 450,
    't_juego_min': 60
}

sensibilidad = {}
ranges = {
    'pasos_dia':         np.linspace(500,  12000, 100),
    'mvpa_min_sem':      np.linspace(0,    420,   100),
    'prec_memoria_pct':  np.linspace(0,    100,   100),
    'tiempo_reaccion_ms':np.linspace(150,  1000,  100),
}
labels_sens = {
    'pasos_dia':          'Steps/day',
    'mvpa_min_sem':       'MVPA (min/week)',
    'prec_memoria_pct':   'Memory performance (%)',
    'tiempo_reaccion_ms': 'Reaction time (ms)',
}
colors_sens = [C1, C3, C4, C2]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6))
axes = axes.flatten()

for ax, (var, rng), col_c in zip(axes, ranges.items(), colors_sens):
    ic_lsg_vals = []
    for v in rng:
        row = baseline.copy()
        row[var] = v
        tmp = pd.DataFrame([row])
        tmp = normalizar_indicadores(tmp)
        ic_lsg_vals.append(tmp['IC_LSG'].values[0])

    # Normalizar rango al eje x para comparabilidad
    ax.plot(rng, ic_lsg_vals, color=col_c, linewidth=1.6)
    ax.axvline(baseline[var], color='black', linewidth=0.8,
               linestyle='--', alpha=0.6, label='Baseline')
    ax.axhline(normalizar_indicadores(pd.DataFrame([baseline]))['IC_LSG'].values[0],
               color=GRAY, linewidth=0.7, linestyle=':', alpha=0.7)
    ax.set_xlabel(labels_sens[var], fontsize=7.5)
    ax.set_ylabel('$IC_{LSG}$' if var in ['pasos_dia','prec_memoria_pct'] else '')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6)
    ax.set_title(f'Variation of: {labels_sens[var]}', fontsize=8)

fig.suptitle('Sensitivity analysis of $IC_{LSG}$ (one variable indicator, fixed remainder)',
             fontsize=9, y=1.01)
fig.tight_layout()
fig.savefig('figures/fig_sensibilidad.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('figures/fig_sensibilidad.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("[OK] fig_sensibilidad guardada\n")

# Fig 5: Arquetipos - radar/scatter IC_fis vs IC_ment
fig, ax = plt.subplots(figsize=(3.4, 2.9))

# Nube de N=50
ax.scatter(df['IC_fis'], df['IC_ment'],
           alpha=0.30, s=18, color=GRAY, zorder=2, label=f'Synthetic Profiles ($N={N}$)')

# Arquetipos
markers = ['o','s','^','D','*']
arq_colors = [C1, C2, C3, '#FF8C00', C4]
for _, row in arq.iterrows():
    idx = arq.index.get_loc(_)
    ax.scatter(row['IC_fis'], row['IC_ment'],
               s=90, zorder=5, marker=markers[idx],
               color=arq_colors[idx],
               label=row['arquetipo'], edgecolors='white', linewidths=0.5)

# Zonas de reglas
ax.axvline(0.60, color=C1, linewidth=0.6, linestyle='--', alpha=0.5)
ax.axhline(0.55, color=C3, linewidth=0.6, linestyle='--', alpha=0.5)
ax.axvline(0.40, color=C2, linewidth=0.6, linestyle='--', alpha=0.4)
ax.axhline(0.35, color=C2, linewidth=0.6, linestyle='--', alpha=0.4)

ax.text(0.62, 0.57, 'R1+R2', fontsize=5.5, color=C1, alpha=0.7)
ax.text(0.02, 0.02, 'R4+R5', fontsize=5.5, color=C2, alpha=0.7)

ax.set_xlabel('$IC_{physical}$'); ax.set_ylabel('$IC_{mental}$')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title('Distribution of profiles and archetypes\nin space $IC_{physical}$ and $IC_{mental}$',
             fontsize=8.5)
ax.legend(frameon=False, fontsize=6, loc='upper left',
          ncol=1, markerscale=0.9)
fig.tight_layout()
fig.savefig('figures/fig_arquetipos.pdf',
            bbox_inches='tight', dpi=300)
fig.savefig('figures/fig_arquetipos.png',
            bbox_inches='tight', dpi=300)
plt.close()
print("[OK] fig_arquetipos guardada\n")

# Exportar tabla CSV
df.round(4).to_csv('data/synthetic_N50.csv', index=False)
arq.round(4).to_csv('data/arquetipos.csv', index=False)
print("\n[OK] CSV exportados")
print("\n[OK] Generación completa. Figuras y datos listos para el paper.")
