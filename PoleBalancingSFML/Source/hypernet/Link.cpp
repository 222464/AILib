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

#include <hypernet/Link.h>

#include <hypernet/HyperNet.h>

using namespace hn;

Link::Link(const Config &config, HyperNet &hypernet, int inputOffset, std::mt19937 &generator)
: _inputOffset(inputOffset)
{
	_response.assign(config._linkResponseSize, 0.0f);

	// Initialize memory
	std::uniform_real_distribution<float> distMemory(config._initMemoryMin, config._initMemoryMax);

	_linkMemory.resize(config._linkMemorySize);

	for (size_t i = 0; i < _linkMemory.size(); i++) {
		std::uniform_real_distribution<float> distMemory(std::get<0>(hypernet._linkMemoryInitRange[i]), std::get<1>(hypernet._linkMemoryInitRange[i]));

		_linkMemory[i] = distMemory(generator);
	}
}

void Link::update(const Config &config, float reward, HyperNet &hypernet, std::mt19937 &generator, float activationMultiplier) {
	const Boid &input = hypernet.getBoid(_inputOffset);

	// Set inputs
	size_t inputIndex = 0;

	for (size_t i = 0; i < input.getNumOutputs(); i++)
		hypernet._linkProcessorInputBuffer[inputIndex++] = input.getOutput(i);

	for (size_t i = 0; i < _linkMemory.size(); i++)
		hypernet._linkProcessorInputBuffer[inputIndex++] = _linkMemory[i];

	hypernet._linkProcessorInputBuffer[inputIndex++] = reward;

	// Set additional random input
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	hypernet._linkProcessorInputBuffer[inputIndex++] = dist01(generator);

	hypernet._linkProcessor.process(hypernet._linkProcessorInputBuffer, hypernet._linkProcessorOutputBuffer, activationMultiplier);

	// Gather outputs
	size_t outputIndex = 0;

	for (size_t i = 0; i < _response.size(); i++)
		_response[i] = hypernet._linkProcessorOutputBuffer[outputIndex++] * config._linkResponseScalar;

	for (size_t i = 0; i < _linkMemory.size(); i++)
		_linkMemory[i] = hypernet._linkProcessorOutputBuffer[outputIndex++];
}