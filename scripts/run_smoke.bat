@echo off
REM Validación end-to-end del pipeline en CPU con datos sintéticos (Windows).
REM Ejecuta desde la raíz del proyecto:  scripts\run_smoke.bat
setlocal
set CFG=config\config_smoke.yaml

python scripts\00_make_synthetic_data.py --config %CFG% || goto :err
python scripts\02_preprocess.py --config %CFG% || goto :err
python scripts\03_train_baseline.py --config %CFG% --dataset busi || goto :err
python scripts\03_train_baseline.py --config %CFG% --dataset bus_bra || goto :err
python scripts\04_cross_domain_matrix.py --config %CFG% || goto :err
python scripts\06_generate_report.py --config %CFG% || goto :err

echo.
echo ===== SMOKE OK: revisa results\reports\SUMMARY.md y results\figures\ =====
exit /b 0

:err
echo.
echo ===== SMOKE FALLO (codigo %errorlevel%) =====
exit /b 1
