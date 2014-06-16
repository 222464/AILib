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

#include <hypernet/HyperNet.h>

#include <algorithm>

#include <iostream>

using namespace hn;

HyperNet::HyperNet()
: _connectDisconnectEnabled(false)
{}

void HyperNet::createRandom(const Config &config, int preTrainIterations, float preTrainAlpha, float preTrainMin, float preTrainMax, std::mt19937 &generator, float activationMultiplier) {
	// + 2 is for reward and random uniform. +3 adds an additional input for the number of links to a neuron
	_linkProcessor.createRandom(config._boidNumOutputs + config._linkMemorySize + 2, config._linkResponseSize + config._linkMemorySize,
		config._linkNumHiddenLayers, config._linkHiddenSize, config._initWeightMin, config._initWeightMax, generator);

	_boidFiringProcessor.createRandom(config._linkResponseSize + config._boidMemorySize + 3, config._boidNumOutputs + config._boidMemorySize,
		config._boidFiringNumHiddenLayers, config._boidFiringHiddenSize, config._initWeightMin, config._initWeightMax, generator);

	_boidConnectProcessor.createRandom((config._boidNumOutputs + config._boidMemorySize) * 2 + 2,
		1, config._boidConnectNumHiddenLayers, config._boidConnectNumHiddenLayers, config._initWeightMin, config._initWeightMax, generator);

	_boidDisconnectProcessor.createRandom((config._boidNumOutputs + config._boidMemorySize) * 2 + config._linkResponseSize + config._linkMemorySize + 2,
		1, config._boidConnectNumHiddenLayers, config._boidConnectNumHiddenLayers, config._initWeightMin, config._initWeightMax, generator);

	_encoder.createRandom(config._numOutputsPerGroup + config._encoderMemorySize, config._linkResponseSize + config._encoderMemorySize,
		config._encoderNumHiddenLayers, config._encoderNumPerHidden, config._initWeightMin, config._initWeightMax, generator);

	_decoder.createRandom(config._boidNumOutputs + config._decoderMemorySize, config._numOutputsPerGroup + config._decoderMemorySize,
		config._decoderNumHiddenLayers, config._decoderNumPerHidden, config._initWeightMin, config._initWeightMax, generator);

	// Init memory ranges
	std::uniform_real_distribution<float> memoryInitDist(config._initMemoryMin, config._initMemoryMax);

	_linkMemoryInitRange.resize(config._linkMemorySize);

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++) {
		_linkMemoryInitRange[i] = std::make_tuple(memoryInitDist(generator), memoryInitDist(generator));

		if (std::get<0>(_linkMemoryInitRange[i]) > std::get<1>(_linkMemoryInitRange[i])) {
			float average = (std::get<0>(_linkMemoryInitRange[i]) + std::get<1>(_linkMemoryInitRange[i])) * 0.5f;

			_linkMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_boidMemoryInitRange.resize(config._boidMemorySize);

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++) {
		_boidMemoryInitRange[i] = std::make_tuple(memoryInitDist(generator), memoryInitDist(generator));

		if (std::get<0>(_boidMemoryInitRange[i]) > std::get<1>(_boidMemoryInitRange[i])) {
			float average = (std::get<0>(_boidMemoryInitRange[i]) + std::get<1>(_boidMemoryInitRange[i])) * 0.5f;

			_boidMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_encoderMemoryInitRange.resize(config._encoderMemorySize);

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++) {
		_encoderMemoryInitRange[i] = std::make_tuple(memoryInitDist(generator), memoryInitDist(generator));

		if (std::get<0>(_encoderMemoryInitRange[i]) > std::get<1>(_encoderMemoryInitRange[i])) {
			float average = (std::get<0>(_encoderMemoryInitRange[i]) + std::get<1>(_encoderMemoryInitRange[i])) * 0.5f;

			_encoderMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_decoderMemoryInitRange.resize(config._decoderMemorySize);

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++) {
		_decoderMemoryInitRange[i] = std::make_tuple(memoryInitDist(generator), memoryInitDist(generator));

		if (std::get<0>(_decoderMemoryInitRange[i]) > std::get<1>(_decoderMemoryInitRange[i])) {
			float average = (std::get<0>(_decoderMemoryInitRange[i]) + std::get<1>(_decoderMemoryInitRange[i])) * 0.5f;

			_decoderMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_linkProcessorInputBuffer.clear();
	_linkProcessorOutputBuffer.clear();
	_boidFiringProcessorInputBuffer.clear();
	_boidFiringProcessorOutputBuffer.clear();
	_boidConnectProcessorInputBuffer.clear();
	_boidConnectProcessorOutputBuffer.clear();
	_boidDisconnectProcessorInputBuffer.clear();
	_boidDisconnectProcessorOutputBuffer.clear();

	_encoderInputBuffer.clear();
	_encoderOutputBuffer.clear();
	_decoderInputBuffer.clear();
	_decoderOutputBuffer.clear();

	_linkProcessorInputBuffer.assign(_linkProcessor.getNumInputs(), 0.0f);
	_linkProcessorOutputBuffer.assign(_linkProcessor.getNumOutputs(), 0.0f);
	_boidFiringProcessorInputBuffer.assign(_boidFiringProcessor.getNumInputs(), 0.0f);
	_boidFiringProcessorOutputBuffer.assign(_boidFiringProcessor.getNumOutputs(), 0.0f);
	_boidConnectProcessorInputBuffer.assign(_boidConnectProcessor.getNumInputs(), 0.0f);
	_boidConnectProcessorOutputBuffer.assign(_boidConnectProcessor.getNumOutputs(), 0.0f);
	_boidDisconnectProcessorInputBuffer.assign(_boidDisconnectProcessor.getNumInputs(), 0.0f);
	_boidDisconnectProcessorOutputBuffer.assign(_boidDisconnectProcessor.getNumOutputs(), 0.0f);

	_encoderInputBuffer.assign(_encoder.getNumInputs(), 0.0f);
	_encoderOutputBuffer.assign(_encoder.getNumOutputs(), 0.0f);
	_decoderInputBuffer.assign(_decoder.getNumInputs(), 0.0f);
	_decoderOutputBuffer.assign(_decoder.getNumOutputs(), 0.0f);

	// Pre-train approximators to not convolute input away
	std::uniform_real_distribution<float> distPreTrain(preTrainMin, preTrainMax);

	for (int i = 0; i < preTrainIterations; i++) {
		for (size_t j = 0; j < _encoderInputBuffer.size(); j++)
			_encoderInputBuffer[j] = distPreTrain(generator);

		float target = 0.0f;

		for (int j = 0; j < config._numInputsPerGroup; j++)
			target += _encoderInputBuffer[j];

		target /= config._numInputsPerGroup;

		std::vector<std::vector<float>> layerOutputs;

		_encoder.process(_encoderInputBuffer, layerOutputs, activationMultiplier);

		for (int j = 0; j < config._linkResponseSize; j++)
			_encoderOutputBuffer[j] = target;

		for (int j = 0; j < config._encoderMemorySize; j++)
			_encoderOutputBuffer[j + config._linkResponseSize] = 0.0f; //_encoderInputBuffer[j + config._numInputsPerGroup];

		_encoder.backpropagate(_encoderInputBuffer, layerOutputs, _encoderOutputBuffer, preTrainAlpha);
	}

	for (int i = 0; i < preTrainIterations; i++) {
		for (size_t j = 0; j < _decoderInputBuffer.size(); j++)
			_decoderInputBuffer[j] = distPreTrain(generator);

		float target = 0.0f;

		for (int j = 0; j < config._linkResponseSize; j++)
			target += _decoderInputBuffer[j];

		target /= config._linkResponseSize;

		std::vector<std::vector<float>> layerOutputs;

		_decoder.process(_decoderInputBuffer, layerOutputs, activationMultiplier);

		for (int j = 0; j < config._numOutputsPerGroup; j++)
			_decoderOutputBuffer[j] = target;

		for (int j = 0; j < config._decoderMemorySize; j++)
			_decoderOutputBuffer[j + config._numOutputsPerGroup] = 0.0f;// _decoderInputBuffer[j + config._linkResponseSize];

		_decoder.backpropagate(_decoderInputBuffer, layerOutputs, _decoderOutputBuffer, preTrainAlpha);
	}

	for (int i = 0; i < preTrainIterations; i++) {
		for (size_t j = 0; j < _linkProcessorInputBuffer.size(); j++)
			_linkProcessorInputBuffer[j] = distPreTrain(generator);

		float target = 0.0f;

		for (int j = 0; j < config._boidNumOutputs; j++)
			target += _linkProcessorInputBuffer[j];

		target /= config._boidNumOutputs;

		std::vector<std::vector<float>> layerOutputs;

		_linkProcessor.process(_linkProcessorInputBuffer, layerOutputs, activationMultiplier);

		for (int j = 0; j < config._linkResponseSize; j++)
			_linkProcessorOutputBuffer[j] = target;

		for (int j = 0; j < config._linkMemorySize; j++)
			_linkProcessorOutputBuffer[j + config._linkResponseSize] = 0.0f; //_linkProcessorInputBuffer[j + config._boidNumOutputs];

		_linkProcessor.backpropagate(_linkProcessorInputBuffer, layerOutputs, _linkProcessorOutputBuffer, preTrainAlpha);
	}

	for (int i = 0; i < preTrainIterations; i++) {
		for (size_t j = 0; j < _boidFiringProcessorInputBuffer.size(); j++)
			_boidFiringProcessorInputBuffer[j] = distPreTrain(generator);

		float target = 0.0f;

		for (int j = 0; j < config._linkResponseSize; j++)
			target += _boidFiringProcessorInputBuffer[j];

		target /= config._linkResponseSize;

		std::vector<std::vector<float>> layerOutputs;

		_boidFiringProcessor.process(_boidFiringProcessorInputBuffer, layerOutputs, activationMultiplier);

		for (int j = 0; j < config._boidNumOutputs; j++)
			_boidFiringProcessorOutputBuffer[j] = target;

		for (int j = 0; j < config._boidMemorySize; j++)
			_boidFiringProcessorOutputBuffer[j + config._boidNumOutputs] = 0.0f; //_boidFiringProcessorInputBuffer[j + config._linkResponseSize];

		_boidFiringProcessor.backpropagate(_boidFiringProcessorInputBuffer, layerOutputs, _boidFiringProcessorOutputBuffer, preTrainAlpha);
	}
}

void HyperNet::createFromParents(const HyperNet &parent1, const HyperNet &parent2,
	float weightAverageChance, float memoryAverageChance,
	std::mt19937 &generator)
{
	_linkProcessor.createFromParents(parent1._linkProcessor, parent2._linkProcessor, weightAverageChance, generator);
	_boidFiringProcessor.createFromParents(parent1._boidFiringProcessor, parent2._boidFiringProcessor, weightAverageChance, generator);
	_boidConnectProcessor.createFromParents(parent1._boidConnectProcessor, parent2._boidConnectProcessor, weightAverageChance, generator);
	_boidDisconnectProcessor.createFromParents(parent1._boidDisconnectProcessor, parent2._boidDisconnectProcessor, weightAverageChance, generator);
	_encoder.createFromParents(parent1._encoder, parent2._encoder, weightAverageChance, generator);
	_decoder.createFromParents(parent1._decoder, parent2._decoder, weightAverageChance, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	_linkMemoryInitRange.resize(parent1._linkMemoryInitRange.size());

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++) {
		if (dist01(generator) < memoryAverageChance)
			_linkMemoryInitRange[i] = std::make_tuple((std::get<0>(parent1._linkMemoryInitRange[i]) + std::get<0>(parent2._linkMemoryInitRange[i])) * 0.5f,
			(std::get<1>(parent1._linkMemoryInitRange[i]) + std::get<1>(parent2._linkMemoryInitRange[i])) * 0.5f);
		else
			_linkMemoryInitRange[i] = dist01(generator) < 0.5f ? parent1._linkMemoryInitRange[i] : parent2._linkMemoryInitRange[i];
	}

	_boidMemoryInitRange.resize(parent1._boidMemoryInitRange.size());

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++) {
		if (dist01(generator) < memoryAverageChance)
			_boidMemoryInitRange[i] = std::make_tuple((std::get<0>(parent1._boidMemoryInitRange[i]) + std::get<0>(parent2._boidMemoryInitRange[i])) * 0.5f,
			(std::get<1>(parent1._boidMemoryInitRange[i]) + std::get<1>(parent2._boidMemoryInitRange[i])) * 0.5f);
		else
			_boidMemoryInitRange[i] = dist01(generator) < 0.5f ? parent1._boidMemoryInitRange[i] : parent2._boidMemoryInitRange[i];
	}

	_encoderMemoryInitRange.resize(parent1._encoderMemoryInitRange.size());

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++) {
		if (dist01(generator) < memoryAverageChance)
			_encoderMemoryInitRange[i] = std::make_tuple((std::get<0>(parent1._encoderMemoryInitRange[i]) + std::get<0>(parent2._encoderMemoryInitRange[i])) * 0.5f,
			(std::get<1>(parent1._encoderMemoryInitRange[i]) + std::get<1>(parent2._encoderMemoryInitRange[i])) * 0.5f);
		else
			_encoderMemoryInitRange[i] = dist01(generator) < 0.5f ? parent1._encoderMemoryInitRange[i] : parent2._encoderMemoryInitRange[i];
	}

	_decoderMemoryInitRange.resize(parent1._decoderMemoryInitRange.size());

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++) {
		if (dist01(generator) < memoryAverageChance)
			_decoderMemoryInitRange[i] = std::make_tuple((std::get<0>(parent1._decoderMemoryInitRange[i]) + std::get<0>(parent2._decoderMemoryInitRange[i])) * 0.5f,
			(std::get<1>(parent1._decoderMemoryInitRange[i]) + std::get<1>(parent2._decoderMemoryInitRange[i])) * 0.5f);
		else
			_decoderMemoryInitRange[i] = dist01(generator) < 0.5f ? parent1._decoderMemoryInitRange[i] : parent2._decoderMemoryInitRange[i];
	}

	_linkProcessorInputBuffer.clear();
	_linkProcessorOutputBuffer.clear();
	_boidFiringProcessorInputBuffer.clear();
	_boidFiringProcessorOutputBuffer.clear();
	_boidConnectProcessorInputBuffer.clear();
	_boidConnectProcessorOutputBuffer.clear();
	_boidDisconnectProcessorInputBuffer.clear();
	_boidDisconnectProcessorOutputBuffer.clear();

	_encoderInputBuffer.clear();
	_encoderOutputBuffer.clear();
	_decoderInputBuffer.clear();
	_decoderOutputBuffer.clear();

	_linkProcessorInputBuffer.assign(_linkProcessor.getNumInputs(), 0.0f);
	_linkProcessorOutputBuffer.assign(_linkProcessor.getNumOutputs(), 0.0f);
	_boidFiringProcessorInputBuffer.assign(_boidFiringProcessor.getNumInputs(), 0.0f);
	_boidFiringProcessorOutputBuffer.assign(_boidFiringProcessor.getNumOutputs(), 0.0f);
	_boidConnectProcessorInputBuffer.assign(_boidConnectProcessor.getNumInputs(), 0.0f);
	_boidConnectProcessorOutputBuffer.assign(_boidConnectProcessor.getNumOutputs(), 0.0f);
	_boidDisconnectProcessorInputBuffer.assign(_boidDisconnectProcessor.getNumInputs(), 0.0f);
	_boidDisconnectProcessorOutputBuffer.assign(_boidDisconnectProcessor.getNumOutputs(), 0.0f);

	_encoderInputBuffer.assign(_encoder.getNumInputs(), 0.0f);
	_encoderOutputBuffer.assign(_encoder.getNumOutputs(), 0.0f);
	_decoderInputBuffer.assign(_decoder.getNumInputs(), 0.0f);
	_decoderOutputBuffer.assign(_decoder.getNumOutputs(), 0.0f);
}

void HyperNet::mutate(float weightPerturbationChance, float weightPerturbationStdDev,
	float memoryPerturbationChance, float memoryPerturbationStdDev, std::mt19937 &generator)
{
	_linkProcessor.mutate(weightPerturbationChance, weightPerturbationStdDev, generator);
	_boidFiringProcessor.mutate(weightPerturbationChance, weightPerturbationStdDev, generator);
	_boidConnectProcessor.mutate(weightPerturbationChance, weightPerturbationStdDev, generator);
	_boidDisconnectProcessor.mutate(weightPerturbationChance, weightPerturbationStdDev, generator);
	_encoder.mutate(weightPerturbationChance, weightPerturbationStdDev, generator);
	_decoder.mutate(weightPerturbationChance, weightPerturbationStdDev, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
	std::normal_distribution<float> distMemoryPerturbation(0.0f, memoryPerturbationStdDev);

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++)
	if (dist01(generator) < memoryPerturbationChance) {
		std::get<0>(_linkMemoryInitRange[i]) += distMemoryPerturbation(generator);
		std::get<1>(_linkMemoryInitRange[i]) += distMemoryPerturbation(generator);

		if (std::get<0>(_linkMemoryInitRange[i]) > std::get<1>(_linkMemoryInitRange[i])) {
			float average = (std::get<0>(_linkMemoryInitRange[i]) + std::get<1>(_linkMemoryInitRange[i])) * 0.5f;

			_linkMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++)
	if (dist01(generator) < memoryPerturbationChance) {
		std::get<0>(_boidMemoryInitRange[i]) += distMemoryPerturbation(generator);
		std::get<1>(_boidMemoryInitRange[i]) += distMemoryPerturbation(generator);

		if (std::get<0>(_boidMemoryInitRange[i]) > std::get<1>(_boidMemoryInitRange[i])) {
			float average = (std::get<0>(_boidMemoryInitRange[i]) + std::get<1>(_boidMemoryInitRange[i])) * 0.5f;

			_boidMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++)
	if (dist01(generator) < memoryPerturbationChance) {
		std::get<0>(_encoderMemoryInitRange[i]) += distMemoryPerturbation(generator);
		std::get<1>(_encoderMemoryInitRange[i]) += distMemoryPerturbation(generator);

		if (std::get<0>(_encoderMemoryInitRange[i]) > std::get<1>(_encoderMemoryInitRange[i])) {
			float average = (std::get<0>(_encoderMemoryInitRange[i]) + std::get<1>(_encoderMemoryInitRange[i])) * 0.5f;

			_encoderMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++)
	if (dist01(generator) < memoryPerturbationChance) {
		std::get<0>(_decoderMemoryInitRange[i]) += distMemoryPerturbation(generator);
		std::get<1>(_decoderMemoryInitRange[i]) += distMemoryPerturbation(generator);

		if (std::get<0>(_decoderMemoryInitRange[i]) > std::get<1>(_decoderMemoryInitRange[i])) {
			float average = (std::get<0>(_decoderMemoryInitRange[i]) + std::get<1>(_decoderMemoryInitRange[i])) * 0.5f;

			_decoderMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}
}

void HyperNet::createFromWeightsVector(const Config &config, const std::vector<float> &weights) {
	size_t weightIndex = 0;

	// + 2 is for reward and random uniform. +3 adds an additional input for the number of links to a neuron
	weightIndex = _linkProcessor.createFromWeightsVector(config._boidNumOutputs + config._linkMemorySize + 2, config._linkResponseSize + config._linkMemorySize,
		config._linkNumHiddenLayers, config._linkHiddenSize, weights, weightIndex);

	weightIndex = _boidFiringProcessor.createFromWeightsVector(config._linkResponseSize + config._boidMemorySize + 3, config._boidNumOutputs + config._boidMemorySize,
		config._boidFiringNumHiddenLayers, config._boidFiringHiddenSize, weights, weightIndex);

	weightIndex = _boidConnectProcessor.createFromWeightsVector((config._boidNumOutputs + config._boidMemorySize) * 2 + 2,
		1, config._boidConnectNumHiddenLayers, config._boidConnectNumHiddenLayers, weights, weightIndex);

	weightIndex = _boidDisconnectProcessor.createFromWeightsVector((config._boidNumOutputs + config._boidMemorySize) * 2 + config._linkResponseSize + config._linkMemorySize + 2,
		1, config._boidConnectNumHiddenLayers, config._boidConnectNumHiddenLayers, weights, weightIndex);

	weightIndex = _encoder.createFromWeightsVector(config._numOutputsPerGroup + config._encoderMemorySize, config._linkResponseSize + config._encoderMemorySize,
		config._encoderNumHiddenLayers, config._encoderNumPerHidden, weights, weightIndex);

	weightIndex = _decoder.createFromWeightsVector(config._boidNumOutputs + config._decoderMemorySize, config._numOutputsPerGroup + config._decoderMemorySize,
		config._decoderNumHiddenLayers, config._decoderNumPerHidden, weights, weightIndex);

	_linkMemoryInitRange.resize(config._linkMemorySize);

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++) {
		_linkMemoryInitRange[i] = std::make_tuple(weights[weightIndex++], weights[weightIndex++]);

		if (std::get<0>(_linkMemoryInitRange[i]) > std::get<1>(_linkMemoryInitRange[i])) {
			float average = (std::get<0>(_linkMemoryInitRange[i]) + std::get<1>(_linkMemoryInitRange[i])) * 0.5f;

			_linkMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_boidMemoryInitRange.resize(config._boidMemorySize);

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++) {
		_boidMemoryInitRange[i] = std::make_tuple(weights[weightIndex++], weights[weightIndex++]);

		if (std::get<0>(_boidMemoryInitRange[i]) > std::get<1>(_boidMemoryInitRange[i])) {
			float average = (std::get<0>(_boidMemoryInitRange[i]) + std::get<1>(_boidMemoryInitRange[i])) * 0.5f;

			_boidMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_encoderMemoryInitRange.resize(config._encoderMemorySize);

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++) {
		_encoderMemoryInitRange[i] = std::make_tuple(weights[weightIndex++], weights[weightIndex++]);

		if (std::get<0>(_encoderMemoryInitRange[i]) > std::get<1>(_encoderMemoryInitRange[i])) {
			float average = (std::get<0>(_encoderMemoryInitRange[i]) + std::get<1>(_encoderMemoryInitRange[i])) * 0.5f;

			_encoderMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_decoderMemoryInitRange.resize(config._decoderMemorySize);

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++) {
		_decoderMemoryInitRange[i] = std::make_tuple(weights[weightIndex++], weights[weightIndex++]);

		if (std::get<0>(_decoderMemoryInitRange[i]) > std::get<1>(_decoderMemoryInitRange[i])) {
			float average = (std::get<0>(_decoderMemoryInitRange[i]) + std::get<1>(_decoderMemoryInitRange[i])) * 0.5f;

			_decoderMemoryInitRange[i] = std::make_tuple(average, average);
		}
	}

	_linkProcessorInputBuffer.clear();
	_linkProcessorOutputBuffer.clear();
	_boidFiringProcessorInputBuffer.clear();
	_boidFiringProcessorOutputBuffer.clear();
	_boidConnectProcessorInputBuffer.clear();
	_boidConnectProcessorOutputBuffer.clear();
	_boidDisconnectProcessorInputBuffer.clear();
	_boidDisconnectProcessorOutputBuffer.clear();

	_encoderInputBuffer.clear();
	_encoderOutputBuffer.clear();
	_decoderInputBuffer.clear();
	_decoderOutputBuffer.clear();

	_linkProcessorInputBuffer.assign(_linkProcessor.getNumInputs(), 0.0f);
	_linkProcessorOutputBuffer.assign(_linkProcessor.getNumOutputs(), 0.0f);
	_boidFiringProcessorInputBuffer.assign(_boidFiringProcessor.getNumInputs(), 0.0f);
	_boidFiringProcessorOutputBuffer.assign(_boidFiringProcessor.getNumOutputs(), 0.0f);
	_boidConnectProcessorInputBuffer.assign(_boidConnectProcessor.getNumInputs(), 0.0f);
	_boidConnectProcessorOutputBuffer.assign(_boidConnectProcessor.getNumOutputs(), 0.0f);
	_boidDisconnectProcessorInputBuffer.assign(_boidDisconnectProcessor.getNumInputs(), 0.0f);
	_boidDisconnectProcessorOutputBuffer.assign(_boidDisconnectProcessor.getNumOutputs(), 0.0f);

	_encoderInputBuffer.assign(_encoder.getNumInputs(), 0.0f);
	_encoderOutputBuffer.assign(_encoder.getNumOutputs(), 0.0f);
	_decoderInputBuffer.assign(_decoder.getNumInputs(), 0.0f);
	_decoderOutputBuffer.assign(_decoder.getNumOutputs(), 0.0f);
}

void HyperNet::getWeightsVector(std::vector<float> &weights) {
	_linkProcessor.getWeightsVector(weights);
	_boidFiringProcessor.getWeightsVector(weights);
	_boidConnectProcessor.getWeightsVector(weights);
	_boidDisconnectProcessor.getWeightsVector(weights);
	_encoder.getWeightsVector(weights);
	_decoder.getWeightsVector(weights);

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++) {
		weights.push_back(std::get<0>(_linkMemoryInitRange[i]));
		weights.push_back(std::get<1>(_linkMemoryInitRange[i]));
	}

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++) {
		weights.push_back(std::get<0>(_boidMemoryInitRange[i]));
		weights.push_back(std::get<1>(_boidMemoryInitRange[i]));
	}

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++) {
		weights.push_back(std::get<0>(_encoderMemoryInitRange[i]));
		weights.push_back(std::get<1>(_encoderMemoryInitRange[i]));
	}

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++) {
		weights.push_back(std::get<0>(_decoderMemoryInitRange[i]));
		weights.push_back(std::get<1>(_decoderMemoryInitRange[i]));
	}
}

int HyperNet::getLinearCoordinate(const std::vector<int> &coordinates) {
	int linearCoord = 0;

	int coordOffset = 1;

	for (int d = 0; d < _dimensions.size(); d++) {
		linearCoord += coordinates[d] * coordOffset;
		coordOffset *= _dimensions[d];
	}

	return linearCoord;
}

void HyperNet::getMultiDimCoordinatesFromLinear(int linearCoord, std::vector<int> &coordinates) {
	if (coordinates.size() != _dimensions.size())
		coordinates.resize(_dimensions.size());

	int coordOffset = 1;

	for (int d = 0; d < _dimensions.size(); d++) {
		coordinates[d] = (linearCoord / coordOffset) % _dimensions[d];
		coordOffset *= _dimensions[d];
	}
}

void HyperNet::generateNetwork(const Config &config, const std::vector<int> &dimensions, std::mt19937 &generator) {
	_dimensions = dimensions;

	_inputs.clear();
	_outputs.clear();
	_boids.clear();

	_inputs.assign(config._numInputGroups * config._numInputsPerGroup, 0.0f);
	_outputs.assign(config._numOutputGroups * config._numOutputsPerGroup, 0.0f);

	int totalSize = 1;

	for (size_t d = 0; d < _dimensions.size(); d++)
		totalSize *= _dimensions[d];

	_boids.resize(totalSize);

	// Create neurons
	for (size_t i = 0; i < _boids.size(); i++)
		_boids[i].reset(new Boid(config, *this, generator));

	int connectionRadiusi = static_cast<int>(std::ceil(config._boidConnectionRadius));

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// Connect neurons to other neurons in a radius
	for (size_t i = 0; i < _boids.size(); i++) {
		std::vector<int> multiDimCoordinates;

		getMultiDimCoordinatesFromLinear(i, multiDimCoordinates);

		std::vector<int> min(_dimensions.size());
		std::vector<int> max(_dimensions.size());

		for (int d = 0; d < _dimensions.size(); d++) {
			min[d] = std::max<int>(0, multiDimCoordinates[d] - connectionRadiusi);
			max[d] = std::min<int>(_dimensions[d] - 1, multiDimCoordinates[d] + connectionRadiusi - 1);
		}

		std::vector<int> parseCoords = min;

		while (parseCoords != max) {
			int linearCoord = getLinearCoordinate(parseCoords);

			if (linearCoord != i) {
				// Calculate distance for radial neuron culling
				float distance = 0.0f;

				for (int d = 0; d < multiDimCoordinates.size(); d++) {
					float delta = static_cast<float>(multiDimCoordinates[d] - parseCoords[d]);
					distance += delta * delta;
				}

				distance = std::sqrt(distance);

				// If the neuron is within range
				if (distance < config._boidConnectionRadius) {
					if (dist01(generator) < config._initLinkChance) {
						std::shared_ptr<Link> link(new Link(config, *this, linearCoord, generator));

						_boids[i]->_links[link->getInputOffset()] = link;
					}
				}
			}

			// Increment coordinates
			parseCoords[0]++;

			for (int d = 0; d < _dimensions.size() - 1; d++)
			if (parseCoords[d] > max[d]) {
				parseCoords[d] = min[d];
				parseCoords[d + 1]++;
			}
			else
				break;
		}
	}

	// Place inputs on front end of the chunk
	int inputDimSize = _dimensions[0];// static_cast<int>(std::ceilf(std::powf(static_cast<float>(config._numInputGroups), 1.0f / _dimensions.size())));
	int outputDimSize = _dimensions[0];// static_cast<int>(std::ceilf(std::powf(static_cast<float>(config._numOutputGroups), 1.0f / _dimensions.size())));

	_inputIndices.resize(config._numInputGroups);
	_outputIndices.resize(config._numOutputGroups);

	float inputDimSizef = static_cast<float>(inputDimSize);
	float outputDimSizef = static_cast<float>(outputDimSize);

	int inputSideOffset = 0;
	int outputSideOffset = 0;// _dimensions[0] / (config._numOutputGroups + 1);

	// ------------------------------------- Input/Output Gathering -------------------------------------

	int allDimsButLast = 1;

	for (size_t d = 0; d < _dimensions.size() - 1; d++)
		allDimsButLast *= _dimensions[d];

	int boidsPerInput = std::min(config._maxBoidsPerInput, allDimsButLast / config._numInputGroups);
	
	int inputBoidIndex = 0;

	for (int inputIndex = 0; inputIndex < config._numInputGroups; inputIndex++)
	for (int perInputIndex = 0; perInputIndex < boidsPerInput; perInputIndex++)
		_inputIndices[inputIndex].push_back(inputBoidIndex++);

	int boidsPerOutput = std::min(config._maxBoidsPerOutput, allDimsButLast / config._numOutputGroups);

	int outputBoidIndex = _boids.size() - 1;

	for (int outputIndex = 0; outputIndex < config._numOutputGroups; outputIndex++)
	for (int perOutputIndex = 0; perOutputIndex < boidsPerOutput; perOutputIndex++)
		_outputIndices[outputIndex].push_back(outputBoidIndex--);

	// Create encoders
	_encoders.resize(config._numInputGroups);

	for (int i = 0; i < config._numInputGroups; i++)
		_encoders[i].reset(new Encoder(config, *this, generator));

	_decoders.resize(config._numOutputGroups);

	for (int i = 0; i < config._numOutputGroups; i++)
		_decoders[i].reset(new Decoder(config, *this, generator));
}

void HyperNet::generateFeedForward(const Config &config, size_t numHiddenLayers, size_t numBoidsPerHiddenLayer, std::mt19937 &generator) {
	_dimensions.clear();

	_inputs.clear();
	_outputs.clear();
	_boids.clear();

	_inputs.assign(config._numInputGroups * config._numInputsPerGroup, 0.0f);
	_outputs.assign(config._numOutputGroups * config._numOutputsPerGroup, 0.0f);

	// Input layer
	for (size_t i = 0; i < _inputs.size(); i++) {
		std::shared_ptr<Boid> boid(new Boid(config, *this, generator));

		_boids.push_back(boid);
	}

	if (numHiddenLayers > 0) {
		// First hidden layer
		for (size_t i = 0; i < numBoidsPerHiddenLayer; i++) {
			std::shared_ptr<Boid> boid(new Boid(config, *this, generator));

			_boids.push_back(boid);

			for (size_t j = 0; j < _inputs.size(); j++) {
				std::shared_ptr<Link> link(new Link(config, *this, j, generator));

				boid->_links[j] = link;
			}
		}

		// All other hidden layers
		for (size_t l = 1; l < numHiddenLayers; l++) {
			int layerStart = _boids.size();

			for (size_t i = 0; i < numBoidsPerHiddenLayer; i++) {
				std::shared_ptr<Boid> boid(new Boid(config, *this, generator));

				_boids.push_back(boid);

				for (size_t j = 0; j < numBoidsPerHiddenLayer; j++) {
					std::shared_ptr<Link> link(new Link(config, *this, layerStart - numBoidsPerHiddenLayer + j, generator));

					boid->_links[layerStart - numBoidsPerHiddenLayer + j] = link;
				}
			}
		}

		// Output layer
		int layerStart = _boids.size();

		for (size_t i = 0; i < _outputs.size(); i++) {
			std::shared_ptr<Boid> boid(new Boid(config, *this, generator));

			_boids.push_back(boid);

			for (size_t j = 0; j < numBoidsPerHiddenLayer; j++) {
				std::shared_ptr<Link> link(new Link(config, *this, layerStart - numBoidsPerHiddenLayer + j, generator));

				boid->_links[layerStart - numBoidsPerHiddenLayer + j] = link;
			}
		}
	}
	else {
		// Output layer
		for (size_t i = 0; i < _outputs.size(); i++) {
			std::shared_ptr<Boid> boid(new Boid(config, *this, generator));

			_boids.push_back(boid);

			for (size_t j = 0; j < _inputs.size(); j++) {
				std::shared_ptr<Link> link(new Link(config, *this, j, generator));

				boid->_links[j] = link;
			}
		}
	}

	_inputIndices.resize(config._numInputGroups);
	_outputIndices.resize(config._numOutputGroups);

	// Set up decoders/encoders
	int inputBoidIndex = 0;

	for (int inputIndex = 0; inputIndex < config._numInputGroups; inputIndex++)
	for (int perInputIndex = 0; perInputIndex < config._numInputsPerGroup; perInputIndex++)
		_inputIndices[inputIndex].push_back(inputBoidIndex++);

	int outputBoidIndex = _boids.size() - 1;

	for (int outputIndex = 0; outputIndex < config._numOutputGroups; outputIndex++)
	for (int perOutputIndex = 0; perOutputIndex < config._numOutputsPerGroup; perOutputIndex++)
		_outputIndices[outputIndex].push_back(outputBoidIndex--);

	// Create encoders
	_encoders.resize(config._numInputGroups);

	for (int i = 0; i < config._numInputGroups; i++)
		_encoders[i].reset(new Encoder(config, *this, generator));

	_decoders.resize(config._numOutputGroups);

	for (int i = 0; i < config._numOutputGroups; i++)
		_decoders[i].reset(new Decoder(config, *this, generator));
}

void HyperNet::step(const Config &config, float reward, std::mt19937 &generator, int substeps, float activationMultiplier) {
	int connectionRadiusi = static_cast<int>(std::ceil(config._boidConnectionRadius));

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (int s = 0; s < substeps; s++) {
		int inputIndex = 0;

		for (int i = 0; i < config._numInputGroups; i++) {
			// Get processed output from encoder
			std::vector<float> groupInput(config._numInputsPerGroup, 0.0f);

			for (int j = 0; j < config._numInputsPerGroup; j++)
				groupInput[j] = _inputs[inputIndex++];

			_encoders[i]->update(groupInput, *this, activationMultiplier);

			// Distribute output to boids
			for (size_t j = 0; j < _inputIndices[i].size(); j++)
			for (int k = 0; k < config._linkResponseSize; k++)
				_boids[_inputIndices[i][j]]->_inputs[k] = _encoders[i]->getOutput(k) * config._boidOutputScalar;
		}

		// Gather inputs from other boids
		for (size_t b = 0; b < _boids.size(); b++)
			_boids[b]->gatherInput(config, reward, *this, generator, b, activationMultiplier);
		
		// Update boids
		for (size_t b = 0; b < _boids.size(); b++)
			_boids[b]->update(config, reward, *this, generator, b, activationMultiplier);

		int outputIndex = 0;

		for (int i = 0; i < config._numOutputGroups; i++) {
			// Get outputs from boids
			std::vector<float> boidOutputs(config._boidNumOutputs, 0.0f);

			for (size_t j = 0; j < _outputIndices[i].size(); j++)
			for (int k = 0; k < config._boidNumOutputs; k++)
				boidOutputs[k] += _boids[_outputIndices[i][j]]->getOutput(k);

			// Get processed output from decoder
			_decoders[i]->update(boidOutputs, *this, activationMultiplier);

			for (int j = 0; j < config._numOutputsPerGroup; j++)
				_outputs[outputIndex++] = _decoders[i]->getOutput(j);
		}
	}

	if (_connectDisconnectEnabled) {
		// Decide whether to connect/disconnect. Not part of substeps for performance reasons
		for (size_t b = 0; b < _boids.size(); b++) {
			std::vector<int> multiDimCoordinates;

			getMultiDimCoordinatesFromLinear(b, multiDimCoordinates);

			std::vector<int> min(_dimensions.size());
			std::vector<int> max(_dimensions.size());

			for (int d = 0; d < _dimensions.size(); d++) {
				min[d] = std::max<int>(0, multiDimCoordinates[d] - connectionRadiusi);
				max[d] = std::min<int>(_dimensions[d] - 1, multiDimCoordinates[d] + connectionRadiusi - 1);
			}

			std::vector<int> parseCoords = min;

			while (parseCoords != max) {
				int parseCoordsLinear = getLinearCoordinate(parseCoords);

				if (parseCoordsLinear != b) {
					// Calculate distance for radial neuron culling
					float distance = 0.0f;

					for (int d = 0; d < multiDimCoordinates.size(); d++) {
						float delta = static_cast<float>(multiDimCoordinates[d] - parseCoords[d]);
						distance += delta * delta;
					}

					distance = std::sqrt(distance);

					// If the neuron is within range
					if (distance < config._boidConnectionRadius) {
						// Check for connect/disconnect

						// See if the boids are connected or not
						std::unordered_map<int, std::shared_ptr<Link>>::iterator it = _boids[b]->_links.find(parseCoordsLinear);

						if (it == _boids[b]->_links.end()) {
							// Link does not exist, consult connector if a link should be made
							int inputIndex = 0;

							for (int i = 0; i < config._boidNumOutputs; i++)
								_boidConnectProcessorInputBuffer[inputIndex++] = _boids[b]->getOutput(i);

							for (int i = 0; i < config._boidMemorySize; i++)
								_boidConnectProcessorInputBuffer[inputIndex++] = _boids[b]->getMemory(i);

							for (int i = 0; i < config._boidNumOutputs; i++)
								_boidConnectProcessorInputBuffer[inputIndex++] = _boids[parseCoordsLinear]->getOutput(i);

							for (int i = 0; i < config._boidMemorySize; i++)
								_boidConnectProcessorInputBuffer[inputIndex++] = _boids[parseCoordsLinear]->getMemory(i);

							_boidConnectProcessorInputBuffer[inputIndex++] = reward;
							_boidConnectProcessorInputBuffer[inputIndex++] = dist01(generator);

							_boidConnectProcessor.process(_boidConnectProcessorInputBuffer, _boidConnectProcessorOutputBuffer, activationMultiplier);

							if (_boidConnectProcessorOutputBuffer[0] > 1.0f) {
								// Connect
								std::shared_ptr<Link> link(new Link(config, *this, parseCoordsLinear, generator));

								_boids[b]->_links[parseCoordsLinear] = link;
							}
						}
						else {
							// Link exists, check if we should disconnect it
							int inputIndex = 0;

							for (int i = 0; i < config._boidNumOutputs; i++)
								_boidDisconnectProcessorInputBuffer[inputIndex++] = _boids[b]->getOutput(i);

							for (int i = 0; i < config._boidMemorySize; i++)
								_boidDisconnectProcessorInputBuffer[inputIndex++] = _boids[b]->getMemory(i);

							for (int i = 0; i < config._boidNumOutputs; i++)
								_boidDisconnectProcessorInputBuffer[inputIndex++] = _boids[parseCoordsLinear]->getOutput(i);

							for (int i = 0; i < config._boidMemorySize; i++)
								_boidDisconnectProcessorInputBuffer[inputIndex++] = _boids[parseCoordsLinear]->getMemory(i);

							// Add synapse as well
							for (int i = 0; i < config._linkResponseSize; i++)
								_boidDisconnectProcessorInputBuffer[inputIndex++] = it->second->getResponse(i);

							for (int i = 0; i < config._linkMemorySize; i++)
								_boidDisconnectProcessorInputBuffer[inputIndex++] = it->second->getMemory(i);

							_boidDisconnectProcessorInputBuffer[inputIndex++] = reward;
							_boidDisconnectProcessorInputBuffer[inputIndex++] = dist01(generator);

							_boidDisconnectProcessor.process(_boidDisconnectProcessorInputBuffer, _boidDisconnectProcessorOutputBuffer, activationMultiplier);

							if (_boidDisconnectProcessorOutputBuffer[0] > 1.0f) {
								// Disconnect
								_boids[b]->_links.erase(it);
							}
						}
					}
				}

				// Increment coordinates
				parseCoords[0]++;

				for (int d = 0; d < _dimensions.size() - 1; d++)
				if (parseCoords[d] > max[d]) {
					parseCoords[d] = min[d];
					parseCoords[d + 1]++;
				}
				else
					break;
			}
		}
	}
}

void HyperNet::writeToStream(std::ostream &os) const {
	_linkProcessor.writeToStream(os);
	_boidFiringProcessor.writeToStream(os);
	_boidConnectProcessor.writeToStream(os);
	_boidDisconnectProcessor.writeToStream(os);
	_encoder.writeToStream(os);
	_decoder.writeToStream(os);

	os << _linkMemoryInitRange.size() << " ";

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++)
		os << std::get<0>(_linkMemoryInitRange[i]) << " " << std::get<1>(_linkMemoryInitRange[i]) << " ";

	os << std::endl;

	os << _boidMemoryInitRange.size() << " ";

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++)
		os << std::get<0>(_boidMemoryInitRange[i]) << " " << std::get<1>(_boidMemoryInitRange[i]) << " ";

	os << std::endl;

	os << _encoderMemoryInitRange.size() << " ";

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++)
		os << std::get<0>(_encoderMemoryInitRange[i]) << " " << std::get<1>(_encoderMemoryInitRange[i]) << " ";

	os << std::endl;

	os << _decoderMemoryInitRange.size() << " ";

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++)
		os << std::get<0>(_decoderMemoryInitRange[i]) << " " << std::get<1>(_decoderMemoryInitRange[i]) << " ";

	os << std::endl;
}

void HyperNet::readFromStream(std::istream &is) {
	_linkProcessor.readFromStream(is);
	_boidFiringProcessor.readFromStream(is);
	_boidConnectProcessor.readFromStream(is);
	_boidDisconnectProcessor.readFromStream(is);
	_encoder.readFromStream(is);
	_decoder.readFromStream(is);

	size_t linkMemoryInitRangeSize;

	is >> linkMemoryInitRangeSize;

	_linkMemoryInitRange.resize(linkMemoryInitRangeSize);

	for (size_t i = 0; i < _linkMemoryInitRange.size(); i++)
		is >> std::get<0>(_linkMemoryInitRange[i]) >> std::get<1>(_linkMemoryInitRange[i]);

	size_t boidMemoryInitRangeSize;

	is >> boidMemoryInitRangeSize;

	_boidMemoryInitRange.resize(boidMemoryInitRangeSize);

	for (size_t i = 0; i < _boidMemoryInitRange.size(); i++)
		is >> std::get<0>(_boidMemoryInitRange[i]) >> std::get<1>(_boidMemoryInitRange[i]);

	size_t encoderMemoryInitRangeSize;

	is >> encoderMemoryInitRangeSize;

	_encoderMemoryInitRange.resize(encoderMemoryInitRangeSize);

	for (size_t i = 0; i < _encoderMemoryInitRange.size(); i++)
		is >> std::get<0>(_encoderMemoryInitRange[i]) >> std::get<1>(_encoderMemoryInitRange[i]);

	size_t decoderMemoryInitRangeSize;

	is >> decoderMemoryInitRangeSize;

	_decoderMemoryInitRange.resize(decoderMemoryInitRangeSize);

	for (size_t i = 0; i < _decoderMemoryInitRange.size(); i++)
		is >> std::get<0>(_decoderMemoryInitRange[i]) >> std::get<1>(_decoderMemoryInitRange[i]);

	_linkProcessorInputBuffer.clear();
	_linkProcessorOutputBuffer.clear();
	_boidFiringProcessorInputBuffer.clear();
	_boidFiringProcessorOutputBuffer.clear();
	_boidConnectProcessorInputBuffer.clear();
	_boidConnectProcessorOutputBuffer.clear();
	_boidDisconnectProcessorInputBuffer.clear();
	_boidDisconnectProcessorOutputBuffer.clear();

	_encoderInputBuffer.clear();
	_encoderOutputBuffer.clear();
	_decoderInputBuffer.clear();
	_decoderOutputBuffer.clear();

	_linkProcessorInputBuffer.assign(_linkProcessor.getNumInputs(), 0.0f);
	_linkProcessorOutputBuffer.assign(_linkProcessor.getNumOutputs(), 0.0f);
	_boidFiringProcessorInputBuffer.assign(_boidFiringProcessor.getNumInputs(), 0.0f);
	_boidFiringProcessorOutputBuffer.assign(_boidFiringProcessor.getNumOutputs(), 0.0f);
	_boidConnectProcessorInputBuffer.assign(_boidConnectProcessor.getNumInputs(), 0.0f);
	_boidConnectProcessorOutputBuffer.assign(_boidConnectProcessor.getNumOutputs(), 0.0f);
	_boidDisconnectProcessorInputBuffer.assign(_boidDisconnectProcessor.getNumInputs(), 0.0f);
	_boidDisconnectProcessorOutputBuffer.assign(_boidDisconnectProcessor.getNumOutputs(), 0.0f);

	_encoderInputBuffer.assign(_encoder.getNumInputs(), 0.0f);
	_encoderOutputBuffer.assign(_encoder.getNumOutputs(), 0.0f);
	_decoderInputBuffer.assign(_decoder.getNumInputs(), 0.0f);
	_decoderOutputBuffer.assign(_decoder.getNumOutputs(), 0.0f);
}