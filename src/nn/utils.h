#pragma once

namespace nn {

	double *random(size_t elementSize);

	double squareError(double d1, double d2);

	double sigmoid(double d);

	double sigmoidDerivation(double d);

	double relu(double d);

	double reluDerivation(double d);
};

