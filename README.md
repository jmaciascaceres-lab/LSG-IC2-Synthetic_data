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

## Fuentes

- [1] González-Ibáñez, R., Macías-Cáceres, J., Villalta-Paucar, M. (2025). LifeSync-Games: A Technical Note on a Novel Framework for Video Game Development. 2025 44th International Conference of the Chilean Computer Science Society (SCCC), Valparaiso, Chile, pp. 1-4, doi: 10.1109/SCCC67219.2025.11420722.<br>
- [2] González-Ibáñez R., Macías-Cáceres J., Villalta-Paucar M., (2025). LifeSync-Games: Toward a Video Game Paradigm for Promoting Responsible Gaming and Human Development. arXiv preprint: 2510.19691 [cs.HC].<br>
- [3] Macías-Cáceres J., Gutiérrez-Vela F., Paderewski-Rodriguez P., González-Ibáñez, R., (2026). LifeSync-Games: Signal-Driven Pervasive Game Design: The LifeSync-Games Framework as a Player Experience Integration Layer. arXiv preprint: 2609.03169 [cs.HC].

