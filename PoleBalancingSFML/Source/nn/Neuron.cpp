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

#include <nn/Neuron.h>

#include <numeric>
#include <algorithm>
#include <iostream>

using namespace nn;

void Neuron::activate(float activationMultiplier, float outputTraceDecay) {
	_output = sigmoid(activationMultiplier * std::accumulate(_synapses.begin(), _synapses.end(), _bias,
		[](float sum, const Synapse &synapse) -> float { return sum + synapse._weight * synapse._pInput->_output; }
	));

	_outputTrace += (2.0f * _output - 1.0f - _outputTrace) * outputTraceDecay;
}

void Neuron::activateTraceless(float activationMultiplier) {
	_output = sigmoid(activationMultiplier * std::accumulate(_synapses.begin(), _synapses.end(), _bias,
		[](float sum, const Synapse &synapse) -> float { return sum + synapse._weight * synapse._pInput->_output; }
	));
}

void Neuron::activateAndReinforce(float activationMultiplier, float outputTraceDecay, float weightTraceDecay, float error) {
	_output = sigmoid(activationMultiplier * std::accumulate(_synapses.begin(), _synapses.end(), _bias,
		[](float sum, const Synapse &synapse) -> float { return sum + synapse._weight * synapse._pInput->_output; }
	));

	_outputTrace += (2.0f * _output - 1.0f - _outputTrace) * outputTraceDecay;

	for (Synapse &synapse : _synapses) {
		synapse._weight += error * synapse._trace;
		synapse._trace += -weightTraceDecay * synapse._trace + (2.0f * _output - 1.0f) * (synapse._pInput->_output);
	}

	_bias += error * _biasTrace;
	_biasTrace += -weightTraceDecay * _biasTrace + 2.0f * _output - 1.0f;
}

void Neuron::activateAndReinforceTraceless(float activationMultiplier, float error) {
	_output = sigmoid(activationMultiplier * std::accumulate(_synapses.begin(), _synapses.end(), _bias,
		[](float sum, const Synapse &synapse) -> float { return sum + synapse._weight * synapse._pInput->_output; }
	));

	for (Synapse &synapse : _synapses)
		synapse._weight += error * (2.0f * _output - 1.0f) * synapse._pInput->_output;

	_bias += error * (2.0f * _output - 1.0f);
}

void Neuron::activateLinear(float activationMultiplier) {
	_output = activationMultiplier * std::accumulate(_synapses.begin(), _synapses.end(), _bias,
		[](float sum, const Synapse &synapse) -> float { return sum + synapse._weight * synapse._pInput->_output; }
	);
}

void Neuron::activateArp(float activationMultiplier, float outputTraceDecay, std::mt19937 &generator) {
	// Firing probability
	_outputTrace = sigmoid(activationMultiplier * std::accumulate(_synapses.begin(), _synapses.end(), _bias,
		[](float sum, const Synapse &synapse) -> float { return sum + synapse._weight * synapse._pInput->_output; }
	));

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	_output = dist01(generator) < _outputTrace ? 1.0f : -1.0f;
}

void Neuron::reinforce(float error, float weightTraceDecay) {
	for (Synapse &synapse : _synapses) {
		synapse._weight += error * synapse._trace;
		synapse._trace += -weightTraceDecay * synapse._trace + (2.0f * _output - 1.0f) * (synapse._pInput->_output);
	}

	_bias += error * _biasTrace;
	_biasTrace += -weightTraceDecay * _biasTrace + 2.0f * _output - 1.0f;
}

void Neuron::reinforceTraceless(float error) {
	for (Synapse &synapse : _synapses)
		synapse._weight += error * (2.0f * _output - 1.0f) * synapse._pInput->_output;

	_bias += error * (2.0f * _output - 1.0f);
}

void Neuron::reinforceArp(float reward, float alpha, float lambda) {
	float expectedOutput = 2.0f * _outputTrace - 1.0f;
	
	for (Synapse &synapse : _synapses)
		synapse._weight += alpha * (reward * (_output - expectedOutput) * synapse._pInput->_output + lambda * (1.0f - reward) * (-_output - expectedOutput) * synapse._pInput->_output);

	_bias += alpha * (reward * (_output - expectedOutput) + lambda * (1.0f - reward) * (-_output - expectedOutput));
}

void Neuron::reinforceArpWithTraces(float reward, float alpha, float lambda, float weightTraceDecay) {
	float expectedOutput = 2.0f * _outputTrace - 1.0f;

	for (Synapse &synapse : _synapses) {
		synapse._trace += -weightTraceDecay * synapse._trace + (_output - expectedOutput) * synapse._pInput->_output;
		synapse._traceAdditional += -weightTraceDecay * synapse._traceAdditional + (-_output - expectedOutput) * synapse._pInput->_output;
		synapse._weight += alpha * (reward * synapse._trace + lambda * (1.0f - reward) * synapse._traceAdditional);
	}

	_biasTrace += -weightTraceDecay * _biasTrace + _output - expectedOutput;
	_biasTraceAdditional += -weightTraceDecay * _biasTraceAdditional - _output - expectedOutput;
	_bias += alpha * (reward * _biasTrace + lambda * (1.0f - reward) * _biasTraceAdditional);
}

void Neuron::reinforceArpMomentum(float reward, float alpha, float lambda, float momentum) {
	float expectedOutput = 2.0f * _outputTrace - 1.0f;

	for (Synapse &synapse : _synapses) {
		float dWeight = alpha * (reward * (_output - expectedOutput) * synapse._pInput->_output + lambda * (1.0f - reward) * (-_output - expectedOutput) * synapse._pInput->_output) + momentum * synapse._trace;
		synapse._weight += dWeight;
		synapse._trace = dWeight;
	}

	float dBias = alpha * (reward * (_output - expectedOutput) + lambda * (1.0f - reward) * (-_output - expectedOutput)) + momentum * _biasTrace;
	_bias += dBias;
	_biasTrace = dBias;
}