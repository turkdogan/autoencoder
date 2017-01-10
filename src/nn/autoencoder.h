#pragma once

class Autoencoder
{
public:
	Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum);
	~Autoencoder();

	void train(double *data) ;
	void test(double *data) ;
	
	// for debugging purposes
	void fullPrint() const;
	void report() const;

private:
	int m_dataDimension; // #of output neurons = #of input neurons
	int m_hiddenDimension;

	double *m_inputValues;
	double *m_hiddenValues;
	double *m_outputValues;
	
	double **m_encoderWeights;
	double **m_decoderWeights;

	double **m_encoderWeightChanges;
	double **m_prevEncoderWeightChanges;
	double **m_decoderWeightChanges;
	double **m_prevDecoderWeightChanges;

	double *m_deltas;

	double m_learningRate;
	double m_momentum;

	double m_error;

	void feedforward() ;
	void backpropagate() const;

};