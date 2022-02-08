default:
	cd dqmc/src/ && python3 -m numpy.f2py -llapack -lblas -c timeflow.f -m timeflow;
	cd dqmc/src/ && python3 -m numpy.f2py -llapack -lblas -c greens.f -m greens;
