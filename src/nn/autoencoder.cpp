#include "autoencoder.h"
#include <iostream>
#include "utils.h"

using namespace std;

Autoencoder::Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum)
{
	m_dataDimension = inputDim;
	m_hiddenDimension = hiddenDim;

	m_inputValues = new double[m_dataDimension];
	m_hiddenValues = new double[m_hiddenDimension];
	m_outputValues = new double[m_dataDimension];

	m_encoderWeights = new double*[m_hiddenDimension];
	for (auto i = 0; i < m_hiddenDimension; i++)
		m_encoderWeights[i] = nn::random(m_dataDimension);

	m_decoderWeights = new double*[m_dataDimension];
	for (auto i = 0; i < m_dataDimension; i++)
		m_decoderWeights[i] = nn::random(m_hiddenDimension);
	
	m_encoderWeightChanges = new double*[m_hiddenDimension];
	m_prevEncoderWeightChanges = new double*[m_hiddenDimension];
	for (auto i = 0; i < m_hiddenDimension; i++)
	{
		m_encoderWeightChanges[i] = new double[m_dataDimension]();
		m_prevEncoderWeightChanges[i] = new double[m_dataDimension]();
	}

	m_decoderWeightChanges = new double*[m_dataDimension];
	m_prevDecoderWeightChanges = new double*[m_dataDimension];
	for (auto i = 0; i < m_dataDimension; i++)
	{
		m_decoderWeightChanges[i] = new double[m_hiddenDimension]();
		m_prevDecoderWeightChanges[i] = new double[m_hiddenDimension]();
	}

	m_deltas = new double[m_dataDimension]();

	m_learningRate = learningRate;
	m_momentum = momentum;

	m_error = 0;
}

void Autoencoder::train(double* data) 
{
	for (auto i = 0; i < m_dataDimension; i++)
		m_inputValues[i] = data[i];

	feedforward();
	backpropagate();
}

void Autoencoder::feedforward() 
{
	// encoder
	for (auto j = 0; j < m_hiddenDimension; j++)
	{
		double total = 0.0;
		for (auto i = 0; i < m_dataDimension; i++)
			total += m_encoderWeights[j][i] * m_inputValues[i];
		m_hiddenValues[j] = nn::sigmoid(total);
	}
	// decoder
	double totalError = 0.0;
	for (auto j = 0; j < m_dataDimension; j++)
	{
		double total = 0.0;
		for (auto i = 0; i < m_hiddenDimension; i++)
		{
			total += m_decoderWeights[j][i] * m_hiddenValues[i];
		}
		m_outputValues[j] = nn::sigmoid(total);
		totalError += nn::squareError(m_outputValues[j], m_inputValues[j]);
	}
	m_error = totalError;

}

void Autoencoder::backpropagate() const
{
	for (auto j = 0; j < m_dataDimension; j++)
	{
		// -(target - out)  partial derivative of error with respect to output
		double dEdO = m_outputValues[j] - m_inputValues[j]; 
		// partial derivative of output with respect to activation
		double dOdN = nn::sigmoidDerivation(m_outputValues[j]);
		m_deltas[j] = dEdO * dOdN;

		for (auto i = 0; i < m_hiddenDimension; i++)
		{
			m_decoderWeightChanges[j][i] = m_hiddenValues[i] * m_deltas[j];
		}
	}
	double *totalErrorDeltas = new double[m_hiddenDimension]();
	for (auto k = 0; k < m_dataDimension; k++)
		for (auto j = 0; j < m_hiddenDimension; j++)
			totalErrorDeltas[j] += m_decoderWeights[k][j] * m_deltas[k];

	for (auto i = 0; i < m_dataDimension; i++)
	{
		for (auto j = 0; j < m_hiddenDimension; j++)
		{
			double dActivation = nn::sigmoidDerivation(m_hiddenValues[j]);
			double change = totalErrorDeltas[j] * m_inputValues[i] * dActivation;
			m_encoderWeightChanges[j][i] = change;
		}
	}

	delete[] totalErrorDeltas;

	for (auto j = 0; j < m_hiddenDimension; j++)
	{
		for (auto i = 0; i < m_dataDimension; i++)
		{
			double prevWeightChange = m_prevEncoderWeightChanges[j][i];
			double weightChange = -m_learningRate * m_encoderWeightChanges[j][i] + m_momentum * prevWeightChange;
			m_prevEncoderWeightChanges[j][i] = weightChange;
			m_encoderWeights[j][i] += weightChange;
		}
	}
	for (auto j = 0; j < m_dataDimension; j++)
	{
		for (auto i = 0; i < m_hiddenDimension; i++)
		{
			double prevWeightChange = m_prevDecoderWeightChanges[j][i];
			double weightChange = -m_learningRate * m_decoderWeightChanges[j][i] + m_momentum * prevWeightChange;
			m_prevDecoderWeightChanges[j][i] = weightChange;
			m_decoderWeights[j][i] += weightChange;
		}
	}
}

void Autoencoder::test(double* data) 
{
	for (auto i = 0; i < m_dataDimension; i++)
		m_inputValues[i] = data[i];

	feedforward();
	for (auto i = 0; i < m_dataDimension; i++)
	{
		cout << m_outputValues[i] << " ";
	}
	cout << endl;
}

Autoencoder::~Autoencoder()
{
	delete[] m_inputValues;
	delete[] m_hiddenValues;
	delete[] m_outputValues;
	
	for (auto i = 0; i < m_hiddenDimension; i++)
	{
		delete[] m_encoderWeights[i];
		delete[] m_encoderWeightChanges[i];
		delete[] m_prevEncoderWeightChanges[i];
	}

	for (auto i = 0; i < m_dataDimension; i++)
	{
		delete[] m_decoderWeights[i];
		delete[] m_decoderWeightChanges[i];
		delete[] m_prevDecoderWeightChanges[i];
	}
	delete[] m_deltas;
}

void Autoencoder::fullPrint() const
{
	cout.precision(10);
	cout << fixed;

	cout << "Encoder:" << endl;
	cout << "Inputs Values: " << endl;
	for (auto i = 0; i < m_dataDimension; i++)
	{
		cout << m_inputValues[i] << " ";
	}
	cout << endl;
	cout << "Encoder Weights: " << endl;
	for (auto i = 0; i < m_dataDimension; i++)
	{
		for (auto j = 0; j < m_hiddenDimension; j++)
		{
			cout << m_encoderWeights[j][i] << ", " << m_encoderWeightChanges[j][i] << ", " << m_prevEncoderWeightChanges[j][i] << endl;
		}
	}
	cout << "Hidden Values: " << endl;
	for (auto i = 0; i < m_hiddenDimension; i++)
	{
		cout << m_hiddenValues[i] << " ";
	}
	cout << endl;
	cout << "Decoder Weights: " << endl;
	for (auto j = 0; j < m_dataDimension; j++)
	{
		for (auto i = 0; i < m_hiddenDimension; i++)
		{
			cout << m_decoderWeights[j][i] << ", " << m_decoderWeightChanges[j][i] << ", " << m_prevDecoderWeightChanges[j][i] << endl;
		}
	}
}

void Autoencoder::report() const
{
	cout.precision(10);
	cout << fixed;
	cout << "Error: " << m_error << endl;
}

