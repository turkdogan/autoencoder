#include "../../nn/autoencoder.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <vector>

using namespace std;

vector<double *> readData(string path) {
	std::vector<double *>data;
	ifstream file(path, ios::in);
	if (!file.is_open()) {
		cerr << "Iris data file could not be read" << endl;
		return data;
	}

	string str;
	while (std::getline(file, str)) {
		std::stringstream ss(str);
		vector<string> tokens;
		double *d = new double[4];
		double *dptr = d;
		double value;
		for (string s; ss >> value;) {
			if (ss.peek() == ',')
				ss.ignore();
			*dptr++ = value;
		}
		data.push_back(d);
	}

	// lets normalize data
	double max = std::numeric_limits<double>::max();
	double min = std::numeric_limits<double>::min();
	double mins[4] = { max, max, max, max};
	double maxes[4] = { min, min, min, min};
	
	for (auto row : data) {
		for (size_t i = 0; i < 4; i++) {
			if (row[i] > maxes[i])
				maxes[i] = row[i];
			if (row[i] < mins[i])
				mins[i] = row[i];
		}
	}
	for (auto row : data) {
		for (size_t i = 0; i < 4; i++) {
			row[i] = (row[i] - mins[i]) / (maxes[i] - mins[i]);
		}
	}
	return data;
}

int main(int argc, char **args) {
	srand(time(NULL));

	int trainingSize = 100;
	vector<double *>data = readData("iris.data");
	std::random_shuffle(data.begin(), data.end());
	Autoencoder *nn = new Autoencoder(4, 20, 0.25, 0.9);
	auto epochs = 100000;
	for (auto i = 0; i < epochs; i++) {
		for (auto count = 0; count < trainingSize; count++) {
			double *row = data.at(count);
			double input[4] = {row[0], row[1], row[2], row[3]};
			nn->train(input);
		}
		if (i % 10000 == 0) {
			nn->report();
		}
	}
	for (auto count = trainingSize; count < data.size(); count++) {
		double *row = data.at(count);
		double input[4] = {row[0], row[1], row[2], row[3]};
		cout << "Expected output " << input[0] << "," <<input[1] <<"," <<input[2] <<","<<input[3] << endl;
		nn->test(input);
	}
	delete nn;
	return 0;
}