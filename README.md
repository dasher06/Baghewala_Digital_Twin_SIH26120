# Baghewala Field Well-to-Surface Digital Twin — SIH Prototype

**SIH 2026 | Problem Statement 26120**  
*Digital Twin for Well-to-Surface Optimization of Cyclic Steam Stimulation (CSS) and Sucker Rod Pump (SRP) Operations for Heavy Oil Wells of Baghewala Field.*

## What is included now

This prototype contains the complete demonstration flow:

1. **Models 1–4** — thermal, production, SRP performance and failure-risk CNN.
2. **Model 5** — constrained joint CSS + SRP optimizer.
3. **Multi-well field layer** — 52 drilled wells, 33 operational wells and 19 non-producing wells (synthetic prototype records).
4. **Field overview** — schematic field map and operational-well status table.
5. **Well selector** — choose any operational well and run the same Digital Twin engine using that well's own completion/operating parameters.
6. **3D well visualization** — interactive schematic wellbore showing surface/SRP, rod string, pump and reservoir; model outputs are displayed beside it.
7. **What-if simulator** — test CSS + SRP changes for the selected well.
8. **Joint optimization** — Model 5 searches feasible settings for the selected well.

> **Data warning:** everything under `data/field/` in this package is synthetic. The 52 well count, 33 operational status distribution, field positions and operating values are prototype placeholders. They are NOT actual Baghewala field measurements or well coordinates.

## Architecture

```text
                         BAGHEWALA FIELD
                                |
                         52 well records
                                |
                     +----------+----------+
                     |      DATA LAYER     |
                     | wells.csv           |
                     | CSS history         |
                     | production history  |
                     | SRP history         |
                     +----------+----------+
                                |
                       SELECT OPERATIONAL WELL
                                |
                       WELL DIGITAL TWIN STATE
                                |
                +---------------+---------------+
                |               |               |
             Model 1         Model 2         Model 3
             Thermal       Production          SRP
                |               |               |
                +---------------+---------------+
                                |
                           Model 4 CNN
                         Failure / Risk
                                |
                           Model 5 Optimizer
                                |
              +-----------------+-----------------+
              |                 |                 |
          What-if          Recommendation      3D view
```

The **3D view is a visualization layer**, not a replacement for the numerical/ML Digital Twin models.

## Folder structure

```text
Baghewala_Digital_Twin/
├── app/
├── data/
│   ├── field/
│   │   ├── wells.csv
│   │   ├── operating_data.csv
│   │   ├── css_history.csv
│   │   ├── production_history.csv
│   │   ├── srp_history.csv
│   │   ├── DATA_MANIFEST.csv
│   │   └── generate_field_data.py
│   ├── real/                 # future real/anonymized field data
│   └── synthetic/            # Models 1–4 synthetic training data
├── physics/
├── inference/
├── models/
├── optimizer/
├── training/
├── tests/
├── dashboard.py
├── requirements.txt
├── .gitignore
└── run_demo.bat
```

## Run on Windows

Open the project in VS Code and open a terminal in the project root.

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Then open `http://localhost:8501`.

If the virtual environment already exists, only run:

```cmd
venv\Scripts\activate
streamlit run dashboard.py
```

## Regenerate the synthetic 52-well field layer

```cmd
python data/field/generate_field_data.py
```

This recreates:

- 52 well master records
- 33 operational / 19 non-producing status records
- current operating conditions for 33 operational wells
- CSS history
- production history
- SRP history

All are explicitly synthetic.

## Models 1–4

The existing training scripts and trained artifacts remain unchanged. They can be regenerated from the original synthetic datasets with:

```cmd
python data/synthetic/generate_thermal_data.py
python data/synthetic/generate_production_data.py
python data/synthetic/generate_srp_data.py
python data/synthetic/generate_dynacard_data.py

python training/train_thermal.py
python training/train_production.py
python training/train_srp.py
python training/train_failure.py
```

Sanity checks:

```cmd
python tests/test_thermal.py
python tests/test_production.py
python tests/test_srp.py
python tests/test_failure.py
python tests/test_integration.py
```

## Replacing synthetic data with real Baghewala data

The field layer is deliberately separated from the model code. The intended production flow is:

```text
SCADA / historian / engineering databases
                |
          data preprocessing
                |
       standardized well_id + timestamp
                |
        field data tables
                |
           Models 1–4
                |
            Model 5
                |
           Dashboard
```

The real data should be mapped to the schemas in `data/field/DATA_MANIFEST.csv`. Keep a stable `well_id` and timestamp across CSS, production and SRP records. Real well coordinates can replace the synthetic `field_x_km` / `field_y_km` fields later.

### Recommended real data categories

- **CSS:** steam volume/rate, pressure, temperature, quality, injection duration, soak duration and cycle number.
- **Thermal/PVT:** reservoir or well temperature, pressure, viscosity and API gravity.
- **Production:** timestamped oil rate, liquid rate and water cut.
- **SRP:** SPM, stroke, VFD, fluid level, pump depth, rod string and tubing configuration.
- **Failure:** raw dynamometer/load-vs-position cards plus maintenance/failure labels.

For Model 4, real labeled dynamometer cards are especially valuable because the current CNN is trained on synthetic cards.

## Important prototype limitations

- The thermal backbone is a simplified approximation, not a certified reservoir simulator.
- The production model is a simplified IPR/decline representation.
- The SRP backbone is a simplified API RP 11L-style approximation, not a certified API RP 11L design calculation.
- The failure cards are synthetic; the CNN is a prototype anomaly/failure classifier.
- Model 5 is a dependency-light constrained random-search optimizer, not an operational control system.
- The 3D well is schematic and not to scale.
- No recommendation should be applied to an actual well without real field data, engineering constraints and domain validation.
