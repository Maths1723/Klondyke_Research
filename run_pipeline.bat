@echo off
REM Full analysis from the YGS map-service export in raw\ (ycgc_layer0-3_*.json).
REM Run setup_venv.bat once first. Total run time: a few hours on a laptop.
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
%PY% scripts\01b_ingest_service.py || goto :err
%PY% scripts\05_dominion_real.py --variants base,drill,zero,full --cv || goto :err
%PY% scripts\05_dominion_real.py --variants base,drillfix --cv --cv-variant drillfix --out-suffix _drillfix || goto :err
%PY% scripts\06_dominion_targets.py || goto :err
%PY% scripts\06_dominion_targets.py --strip || goto :err
%PY% scripts\06_dominion_targets.py --sd-hole 0.7 || goto :err
%PY% scripts\07_sulphur_dredge_check.py || goto :err
%PY% scripts\07_sulphur_dredge_check.py --price35 || goto :err
%PY% scripts\07_sulphur_dredge_check.py --radius 40 || goto :err
%PY% scripts\08_error_bound.py || goto :err
%PY% scripts\09_meff_from_production.py || goto :err
%PY% scripts\10_regional_grain.py || goto :err
%PY% scripts\10_regional_grain.py --sd-hole 0.7 || goto :err
%PY% scripts\11_twin_holes.py || goto :err
%PY% scripts\12_grain_profiles.py || goto :err
%PY% scripts\13_identifiability.py || goto :err
%PY% scripts\14_mcmc_check.py --cases synthetic,dominion || goto :err
%PY% scripts\14_mcmc_check.py --cases dominion --start dispersed --seed 17 || goto :err
%PY% scripts\15_design.py || goto :err
%PY% scripts\16_robustness_checks.py || goto :err
%PY% scripts\17_twin_uncertainty.py || goto :err
for %%s in (11 12 13) do %PY% scripts\run_physical_benchmark.py --seed %%s || goto :err
%PY% scripts\18_baseline_check.py || goto :err
%PY% paper\make_figures.py || goto :err
%PY% paper\check_numbers.py || goto :err
echo Done. See reports\ and paper\
goto :eof
:err
echo Pipeline stopped - see the message above.
exit /b 1
