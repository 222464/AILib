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

#include <hypernet/Boid.h>

#include <hypernet/HyperNet.h>

#include <assert.h>

using namespace hn;

Boid::Boid(const Config &config, HyperNet &hypernet, std::mt19937 &generator) {
	_inputs.assign(config._linkResponseSize, 0.0f);

	_outputs.assign(config._boidNumOutputs, 0.0f);

	// Initialize memory
	_boidMemory.resize(config._boidMemorySize);

	for (size_t i = 0; i < _boidMemory.size(); i++) {
		std::uniform_real_distribution<float> distMemory(std::get<0>(hypernet._boidMemoryInitRange[i]), std::get<1>(hypernet._boidMemoryInitRange[i]));

		_boidMemory[i] = distMemory(generator);
	}
}

void Boid::gatherInput(const Config &config, float reward, class HyperNet &hypernet, std::mt19937 &generator, int myInputOffset, float activationMultiplier) {
	for (std::unordered_map<int, std::shared_ptr<Link>>::iterator it = _links.begin(); it != _links.end(); it++) {
		assert(it->second->getInputOffset() != myInputOffset);

		it->second->update(config, reward, hypernet, generator, activationMultiplier);

		for (size_t j = 0; j < _inputs.size(); j++)
			_inputs[j] += it->second->getResponse(j);
	}
}

void Boid::update(const Config &config, float reward, class HyperNet &hypernet, std::mt19937 &generator, int myInputOffset, float activationMultiplier) {
	// Set inputs
	size_t inputIndex = 0;

	for (size_t i = 0; i < _inputs.size(); i++) {
		hypernet._boidFiringProcessorInputBuffer[inputIndex++] = _inputs[i];

		// Zero input
		_inputs[i] = 0.0f;
	}

	for (size_t i = 0; i < _boidMemory.size(); i++)
		hypernet._boidFiringProcessorInputBuffer[inputIndex++] = _boidMemory[i];

	hypernet._boidFiringProcessorInputBuffer[inputIndex++] = reward;

	// Set additional random input
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	hypernet._boidFiringProcessorInputBuffer[inputIndex++] = dist01(generator);

	hypernet._boidFiringProcessorInputBuffer[inputIndex++] = static_cast<float>(_links.size()) * 0.125f;

	hypernet._boidFiringProcessor.process(hypernet._boidFiringProcessorInputBuffer, hypernet._boidFiringProcessorOutputBuffer, activationMultiplier);

	// Gather outputs
	size_t outputIndex = 0;

	for (size_t i = 0; i < _outputs.size(); i++)
		_outputs[i] = hypernet._boidFiringProcessorOutputBuffer[outputIndex++] * config._boidOutputScalar;

	for (size_t i = 0; i < _boidMemory.size(); i++)
		_boidMemory[i] = hypernet._boidFiringProcessorOutputBuffer[outputIndex++];
}