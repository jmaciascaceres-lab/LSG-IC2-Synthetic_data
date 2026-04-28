# IC₂_LSG | Instrucciones de compilación

## Archivos del proyecto

```
LSG-IC2-Synthetic_data/
├── synthetic_data.py
├── data/
│   ├── synthetic_N50.csv
│   ├── arquetipos.csv
├── figures/
│   ├── fig_distribucion.png
│   ├── fig_geom_vs_aritm.png
│   ├── fig_sensibilidad.png
│   ├── fig_arquetipos.png
│   ├── fig_distribucion.pdf
│   ├── fig_geom_vs_aritm.pdf
│   ├── fig_sensibilidad.pdf
│   └── fig_arquetipos.pdf
└── README.md
```

## Generación de datos sintéticos para validación preliminar

### Instalación de librerías
```bash
pip install numpy pandas matplotlib scipy
```

### Para generar datos sintéticos
```bash
python synthetic_data.py
```

