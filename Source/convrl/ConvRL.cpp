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

#include <convrl/ConvRL.h>

#include <iostream>

using namespace deep;

ConvRL::ConvRL()
{}

void ConvRL::createRandom(int inputMapWidth, int inputMapHeight, int inputNumMaps, const std::vector<ConvNet2D::LayerPairDesc> &layerDescs, float convMinWeight, float convMaxWeight, int numOutputs, int ferlNumHidden, float ferlWeightStdDev, std::mt19937 &generator) {
	_convNet.createRandom(inputMapWidth, inputMapHeight, inputNumMaps, layerDescs, convMinWeight, convMaxWeight, generator);

	_ferl.createRandom(_convNet.getOutputWidth() * _convNet.getOutputHeight() * _convNet.getOutputNumMaps(), numOutputs, ferlNumHidden, ferlWeightStdDev, generator);
	
	_outputs.clear();
	_outputs.assign(numOutputs, 0.0f);
}

void ConvRL::step(float reward, float qAlpha, float gamma, float lambdaGamma, float tauInv,
	float rbmAlpha, int convMaxNumReplaySamples, int convNumReplayIterations,
	int actionSearchIterations, int actionSearchSamples, float actionSearchAlpha,
	float breakChance, float perturbationStdDev,
	int ferlMaxNumReplaySamples, int ferlReplayIterations,
	float gradientAlpha, float gradientMomentum,
	std::mt19937 &generator, std::vector<float> &convBuff)
{
	ReplaySample sample;

	sample._inputs.resize(_convNet.getInputWidth() * _convNet.getInputHeight() * _convNet.getInputNumMaps());

	int inputIndex = 0;

	for (int x = 0; x < _convNet.getInputWidth(); x++)
	for (int y = 0; y < _convNet.getInputHeight(); y++)
	for (int m = 0; m < _convNet.getInputNumMaps(); m++)
		sample._inputs[inputIndex++] = _convNet.getInput(x, y, m);

	_replayChain.push_front(sample);

	while (_replayChain.size() > convMaxNumReplaySamples)
		_replayChain.pop_back();

	// Train on samples
	std::vector<ReplaySample*> pSamples(_replayChain.size());

	int index = 0;

	for (std::list<ReplaySample>::iterator it = _replayChain.begin(); it != _replayChain.end(); it++)
		pSamples[index++] = &(*it);

	std::uniform_int_distribution<int> convReplayDist(0, pSamples.size() - 1);

	for (int i = 0; i < convNumReplayIterations; i++) {
		int replayIndex = convReplayDist(generator);

		ReplaySample* pSample = pSamples[replayIndex];

		inputIndex = 0;

		for (int x = 0; x < _convNet.getInputWidth(); x++)
		for (int y = 0; y < _convNet.getInputHeight(); y++)
		for (int m = 0; m < _convNet.getInputNumMaps(); m++)
			_convNet.setInput(x, y, m, pSample->_inputs[inputIndex++]);

		_convNet.activateAndLearn(rbmAlpha, generator);
	}

	inputIndex = 0;

	for (int x = 0; x < _convNet.getInputWidth(); x++)
	for (int y = 0; y < _convNet.getInputHeight(); y++)
	for (int m = 0; m < _convNet.getInputNumMaps(); m++)
		_convNet.setInput(x, y, m, sample._inputs[inputIndex++]);

	_convNet.activate();

	std::vector<float> condensedInputf(_convNet.getOutputWidth() * _convNet.getOutputHeight() * _convNet.getOutputNumMaps());

	int outputIndex = 0;

	for (int x = 0; x < _convNet.getOutputWidth(); x++)
	for (int y = 0; y < _convNet.getOutputHeight(); y++)
	for (int m = 0; m < _convNet.getOutputNumMaps(); m++)
		condensedInputf[outputIndex++] = _convNet.getOutput(x, y, m);

	convBuff = condensedInputf;

	_ferl.step(condensedInputf, _outputs,
		reward, qAlpha, gamma, lambdaGamma, tauInv,
		actionSearchIterations, actionSearchSamples, actionSearchAlpha,
		breakChance, perturbationStdDev,
		ferlMaxNumReplaySamples, ferlReplayIterations, gradientAlpha, gradientMomentum,
		generator);
}