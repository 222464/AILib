#include <deep/DeepSOMNet.h>

#include <algorithm>

#include <iostream>

using namespace deep;

void DeepSOMNet::createRandom(int numInputs, int numOutputs, int SOMChainSize, int startNumDimensions, int numDimensionsLessPerSOM, int SOMSize, int FANumHidden, int FANumPerHidden, float minSOMWeight, float maxSOMWeight, float minFAWeight, float maxFAWeight, std::mt19937 &generator) {
	_SOMChain.resize(SOMChainSize);

	int numSOMInputs = startNumDimensions;

	for (int i = 0; i < SOMChainSize; i++) {
		int numSOMOutputs = std::max(1, numSOMInputs - numDimensionsLessPerSOM);

		if (i == 0)
			_SOMChain[i].createRandom(numInputs, numSOMOutputs, SOMSize, minSOMWeight, maxSOMWeight, generator);
		else
			_SOMChain[i].createRandom(numSOMInputs, numSOMOutputs, SOMSize, minSOMWeight, maxSOMWeight, generator);

		numSOMInputs = numSOMOutputs;
	}

	_fa.createRandom(numSOMInputs, numOutputs, FANumHidden, FANumPerHidden, minFAWeight, maxFAWeight, generator);

	_inputs.assign(numInputs, 0.0f);
	_outputs.assign(numOutputs, 0.0f);
}

void DeepSOMNet::activate() {
	std::vector<float> inputs = _inputs;

	for (size_t i = 0; i < _SOMChain.size(); i++) {
		DSOM::SOMCoordsReal real = _SOMChain[i].getBestMatchingUnitReal(inputs, 0.5f);

		inputs = real._coords;
	}

	_fa.process(inputs, _outputs, 1.0f);
}

void DeepSOMNet::activateAndLearn(const std::vector<float> &targets, float FAAlpha) {
	std::vector<float> inputs = _inputs;

	for (size_t i = 0; i < _SOMChain.size(); i++) {
		DSOM::SOMCoords discrete = _SOMChain[i].getBestMatchingUnit(inputs);
		DSOM::SOMCoordsReal real = _SOMChain[i].getBestMatchingUnitReal(inputs, 0.5f);

		_SOMChain[i].updateNeighborhood(discrete, inputs);

		inputs = real._coords;
	}

	// Normalize
	float nMult = 1.0f / _SOMChain.back().getDimensionSize();

	for (size_t i = 0; i < inputs.size(); i++)
		inputs[i] *= nMult;

	_fa.process(inputs, _layerOutputs, 1.0f);

	_outputs = _layerOutputs.back();

	_fa.backpropagate(inputs, _layerOutputs, targets, FAAlpha);
}