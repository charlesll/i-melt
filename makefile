prepare_data:
	python3 ./src/Dataset_preparation.py

see_data:
	python3 ./src/Dataset_visualization.py

train_one:
	python3 ./src/Training_single.py

train_candidates:
	python3 ./src/Training_candidates.py

bo:
	python3 ./src/bayesian_optim.py > ./model/bo.log

ray:
	python3 ./src/ray_opt.py > ./model/ray3.log

select:
	python3 ./src/ray_select.py

transfer:
	tar czvf compressed_files.tar.gz ./data/NKCMAS* ./src/* ./makefile
	scp compressed_files.tar.gz lelosq@stellagpu01:/gpfs/users/lelosq/neuravi/

transfer_back:
	scp lelosq@stellagpu01:/gpfs/users/lelosq/neuravi/model/l4_n400_p0.15_GELU_cpfree.pth ~/Sync_me/neuravi_DEV/model/

decompress:
	tar xzvf compressed_files.tar.gz
	