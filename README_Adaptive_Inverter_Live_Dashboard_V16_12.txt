Adaptive Inverter V16.12 — Live Simulation Dashboard
=====================================================

WHAT CHANGED
------------
The Individual Case Explorer no longer selects or approximates from Monte Carlo rows.
Each live run uses the saved V16.8 neural-network surrogate and executes the Grey Wolf
Optimization search at the requested PF, R, ambient temperature and study time.

Live-case sequence:
1. Validate the requested point inside the trained domain.
2. Run GWO: 24 wolves x 40 iterations x 2 repeats.
3. Evaluate the continuous GWO candidate plus all allowed measured-frequency safeguards.
4. Apply the frequency-first AI supervisory policy.
5. If Complete Supervisor is selected, run the AI power search only when no valid
   full-power AI frequency remains at or below 150 C.
6. Evaluate the selected fixed-frequency reference using the same prediction and
   electrothermal pipeline.
7. Calculate the user-selected metrics and the Arrhenius relative thermal-aging ratio.
8. Display the GWO convergence and candidate evidence.

The progress bar follows the actual GWO iterations. No artificial sleep/wait is added.

COLAB
-----
1. Upload Adaptive_Inverter_Live_Dashboard_V16_12_Package.zip to Colab.
2. Upload/open Adaptive_Inverter_Live_Dashboard_V16_12_Colab.ipynb.
3. Run all cells.
4. Open the Gradio public link printed by the final cell.

LOCAL
-----
1. Extract the package.
2. In a terminal inside the extracted folder:
   pip install -r requirements_v1612.txt
   python adaptive_inverter_live_dashboard_v1612.py
3. Open the local Gradio URL.

IMPORTANT
---------
This is a live ANN+GWO+electrothermal inference dashboard, not a MATLAB/Simulink server.
The final Simulink model is still the independent physical validation layer used in the paper.
The live dashboard reproduces the final V16.8 AI decision pipeline without Monte Carlo lookup.

RELATIVE THERMAL LIFETIME
-------------------------
The dashboard uses the Arrhenius ratio:
  L_AI / L_fixed = exp[(Ea/k)*(1/T_AI - 1/T_fixed)]
with temperature in Kelvin and user-selectable Ea (default 0.70 eV).
This is presented as a relative temperature-accelerated aging multiplier, not an absolute
number of years. The metric is not calculated if either compared state reaches 175 C.
