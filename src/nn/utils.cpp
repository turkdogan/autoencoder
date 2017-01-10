#include <random>
#include <algorithm>
#include <iostream>

namespace nn {

	double *random(size_t elementSize) {
		double *result = new double[elementSize];
		for (size_t i = 0; i < elementSize; i++) {
			result[i] = ((double)rand() / (RAND_MAX));
			// result[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
		}
		return result;
	}

	double *randomGaussian(size_t elementSize, double mean, double sigma) {
		std::default_random_engine generator;
		std::normal_distribution<double> distribution(mean, sigma);
		double *result = new double[elementSize];
		for (size_t i = 0; i < elementSize; i++) {
			result[i] = distribution(generator);
		}
		return result;
	}

	double squareError(double d1, double d2) {
		return pow((d1 - d2), 2);
	}

	double sigmoid(double d) {
		return 1.0 / (1.0 + exp(-d));
	}

	double sigmoidDerivation(double d) {
		return d * (1.0 - d);
	}

	double relu(double d) {
		return std::fmax(0, d);
	}

	double reluDerivation(double d) {
		return d >= 0.0 ? 0.0 : 1.0;
	}
};

