# Russian Baseline (Deterministic + 3 Seeds) - Dev and Test
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_dev_blind.txt --out-name author_rus_dev_deterministic.tsv
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_dev_blind.txt --out-name author_rus_dev_s123.tsv --seed 123 --stochastic True
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_dev_blind.txt --out-name author_rus_dev_s456.tsv --seed 456 --stochastic True
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_dev_blind.txt --out-name author_rus_dev_s789.tsv --seed 789 --stochastic True
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_test_blind.txt --out-name author_rus_test_deterministic.tsv
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_test_blind.txt --out-name author_rus_test_s123.tsv --seed 123 --stochastic True
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_test_blind.txt --out-name author_rus_test_s456.tsv --seed 456 --stochastic True
modal run BioNNE-R/modal_app.py::author_baseline_predict --track rus --data-name rus_test_blind.txt --out-name author_rus_test_s789.tsv --seed 789 --stochastic True
