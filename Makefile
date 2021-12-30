default:
	cd dqmc/src/ && python3 -m numpy.f2py -llapack -lblas -c _timeflow.f90 -m _timeflow;
	cd dqmc/src/ && python3 -m numpy.f2py -llapack -lblas -c _greens.f90 -m _greens;
