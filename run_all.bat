@echo off
python -m pip install -r requirements.txt
cd src
python bdf2_temporal_interface.py
python run_additional_semidiscrete_benchmark.py --task all --J 160 --m-min 5 --m-max 9 --timing-N 64,128,256 --repeats 5
python figure_1_concept.py
python figure_2_mechanisms.py
