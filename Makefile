default:
	cd dqmc/src/ && python3 -m numpy.f2py -llapack -lblas -c timeflow.f90 -m timeflow;
	cd dqmc/src/ && python3 -m numpy.f2py -llapack -lblas -c greens.f90 -m greens;
