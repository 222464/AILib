/*
AI Lib
Copyright (C) 2014 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include <nn/QAgent.h>
#include <algorithm>
#include <iostream>

using namespace nn;

QAgent::QAgent()
: _qDegradeAccum(0.0f),
_alpha(1.0f), _gamma(0.95f), _maxFitness(999.0f),
_findMaxQPasses(4), _findMaxQSamples(3), _numBackpropPasses(8),
_qUpdateAlpha(0.01f), _findMaxQAlpha(0.01f), _findMaxQMometum(0.0f),
_momentum(0.0f),
_randInitOutputRange(1.0f),
_numPseudoRehearsalSamples(50), _pseudoRehearsalSampleStdDev(0.75f), _pseudoRehearsalSampleMean(0.0f),
_weightDecay(0.0f), _qDecay(0.0f), _stdDev(0.1f)
{}

void QAgent::createRandom(size_t numInputs, size_t numOutputs,
	size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
	float minWeight, float maxWeight, unsigned long seed)
{
	_numInputs = numInputs;

	std::mt19937 generator;
	generator.seed(seed);

	_qNetwork.createRandom(_numInputs + numOutputs, 1, numHiddenLayers, numNeuronsPerHiddenLayer, minWeight, maxWeight, generator);

	_prevInputs.assign(_qNetwork.getNumInputs(), 0.0f);
	_outputBuffer.assign(numOutputs, 0.0f);

	_outputVelocities.assign(numOutputs, 0.0f);
	_outputOffsets.assign(numOutputs, 0.0f);

	_generator.seed(seed);
}

void QAgent::findMaxQGradient() {
	std::uniform_real_distribution<float> distributionOutputRange(-_randInitOutputRange, _randInitOutputRange);

	std::vector<float> maxQSample(getNumOutputs(), 0.0f);
	float maxQ = -99999.0f;

	for (size_t s = 0; s < _findMaxQSamples; s++) {
		for (size_t i = 0; i < getNumOutputs(); i++)
			_qNetwork.setInput(i + _numInputs, distributionOutputRange(_generator));

		// Momentum for finding Q
		std::vector<float> prevDOutputs(getNumOutputs(), 0.0f);

		for (size_t p = 0; p < _findMaxQPasses; p++) {
			_qNetwork.activateLinearOutputLayer();

			float q = _qNetwork.getOutput(0);

			if (q > maxQ) {
				maxQ = q;

				for (size_t i = 0; i < getNumOutputs(); i++)
					maxQSample[i] = _qNetwork.getInput(i + _numInputs);
			}

			FeedForwardNeuralNetwork::Gradient grad;
			_qNetwork.getGradientLinearOutputLayer(std::vector<float>(1, _maxFitness), grad);

			std::vector<float> inputGradient;
			_qNetwork.getInputGradient(grad, inputGradient);

			for (size_t i = 0; i < getNumOutputs(); i++) {
				size_t index = i + _numInputs;

				float dOutput = inputGradient[index] * _findMaxQAlpha + _findMaxQMometum * prevDOutputs[i];

				float newOutput = _qNetwork.getInput(index) + dOutput;

				if (newOutput > 1.0f)
					newOutput = 1.0f;
				else if (newOutput < -1.0f)
					newOutput = -1.0f;

				_qNetwork.setInput(index, newOutput);

				prevDOutputs[i] = dOutput;
			}
		}

		_qNetwork.activateLinearOutputLayer();

		float q = _qNetwork.getOutput(0);

		if (q > maxQ) {
			maxQ = q;

			for (size_t i = 0; i < getNumOutputs(); i++)
				maxQSample[i] = _qNetwork.getInput(i + _numInputs);
		}
	}

	// Set max sample
	for (size_t i = 0; i < getNumOutputs(); i++)
		_qNetwork.setInput(i + _numInputs, maxQSample[i]);
}

void QAgent::step(float fitness) {
	std::vector<float> currentInputs(_qNetwork.getNumInputs());

	for (size_t i = 0; i < _numInputs; i++)
		currentInputs[i] = _qNetwork.getInput(i);

	// Get pseudorehearsal samples
	std::vector<IOSet> rehearsalSamples(_numPseudoRehearsalSamples);

	std::normal_distribution<float> pseudoRehearsalInputDistribution(_pseudoRehearsalSampleMean, _pseudoRehearsalSampleStdDev);

	for (size_t i = 0; i < _numPseudoRehearsalSamples; i++) {
		// Generate sample
		rehearsalSamples[i]._inputs.resize(_qNetwork.getNumInputs());

		for (size_t j = 0; j < _qNetwork.getNumInputs(); j++) {
			rehearsalSamples[i]._inputs[j] = pseudoRehearsalInputDistribution(_generator);
			_qNetwork.setInput(j, rehearsalSamples[i]._inputs[j]);
		}

		_qNetwork.activateLinearOutputLayer();

		rehearsalSamples[i]._output = _qNetwork.getOutput(0);
	}

	for (size_t i = 0; i < _numInputs; i++)
		_qNetwork.setInput(i, _prevInputs[i]);

	_qNetwork.activateLinearOutputLayer();

	float prevQ = _qNetwork.getOutput(0);

	for (size_t i = 0; i < _numInputs; i++)
		_qNetwork.setInput(i, currentInputs[i]);

	findMaxQGradient();

	for (size_t i = 0; i < _qNetwork.getNumInputs(); i++)
		currentInputs[i] = _qNetwork.getInput(i);

	std::normal_distribution<float> normalDist(0.0f, _stdDev);

	// Store chosen action
	for (size_t i = 0; i < getNumOutputs(); i++)
		_outputBuffer[i] = currentInputs[i + _numInputs] = std::min(1.0f, std::max(-1.0f, _qNetwork.getInput(i + _numInputs))) + normalDist(_generator);

	_qNetwork.activateLinearOutputLayer();

	// Compute new Q
	float nextQ = _qNetwork.getOutput(0);

	//system("pause");

	// Update Q
	float newQ = (1.0f - _alpha) * prevQ + _alpha * (fitness + _gamma * nextQ);

	std::cout << "NQ: " << nextQ << " " << newQ << std::endl;

	for (size_t p = 0; p < _numBackpropPasses; p++) {
		// Train on new sample
		for (size_t i = 0; i < _qNetwork.getNumInputs(); i++)
			_qNetwork.setInput(i, _prevInputs[i]);

		_qNetwork.activateLinearOutputLayer();

		FeedForwardNeuralNetwork::Gradient grad;
		_qNetwork.getGradientLinearOutputLayer(std::vector<float>(1, newQ), grad);

		_qNetwork.moveAlongGradientMomentum(grad, _qUpdateAlpha, _momentum);

		// Train on rehearsal samples
		for (size_t i = 0; i < _numPseudoRehearsalSamples; i++) {
			for (size_t j = 0; j < _qNetwork.getNumInputs(); j++)
				_qNetwork.setInput(j, rehearsalSamples[i]._inputs[j]);

			_qNetwork.activateLinearOutputLayer();

			FeedForwardNeuralNetwork::Gradient grad;
			_qNetwork.getGradientLinearOutputLayer(std::vector<float>(1, rehearsalSamples[i]._output), grad);

			_qNetwork.moveAlongGradientMomentum(grad, _qUpdateAlpha, _momentum);
		}
	}

	_qNetwork.decayWeights(_weightDecay);

	_prevInputs = currentInputs;
}

void QAgent::writeToStream(std::ostream &stream)
{
	stream << _alpha << " " << _gamma << " " << _maxFitness << " " << _qUpdateAlpha << " " << _findMaxQAlpha << " " <<
		_findMaxQMometum << " " << _momentum << " " << _randInitOutputRange << std::endl;

	stream << _findMaxQPasses << " " << _findMaxQSamples << " " << _numBackpropPasses << std::endl;
	stream << _numInputs << std::endl;

	_qNetwork.writeToStream(stream);
}

void QAgent::readFromStream(std::istream &stream)
{
	stream >> _alpha;
	stream >> _gamma;
	stream >> _maxFitness;
	stream >> _qUpdateAlpha;
	stream >> _findMaxQAlpha;
	stream >> _findMaxQMometum;
	stream >> _momentum;
	stream >> _randInitOutputRange;

	stream >> _findMaxQPasses;
	stream >> _findMaxQSamples;
	stream >> _numBackpropPasses;;

	stream >> _numInputs;

	_qNetwork.readFromStream(stream);

	size_t numOutputs = _qNetwork.getNumInputs() - _numInputs;

	_prevInputs.assign(_qNetwork.getNumInputs(), 0.0f);
	_outputBuffer.assign(numOutputs, 0.0f);

	_outputVelocities.assign(numOutputs, 0.0f);
	_outputOffsets.assign(numOutputs, 0.0f);
}