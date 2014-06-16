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

#include <hypernet/Decoder.h>

#include <hypernet/HyperNet.h>

using namespace hn;

Decoder::Decoder(const Config &config, HyperNet &hypernet, std::mt19937 &generator) {
	_outputs.assign(config._numOutputsPerGroup, 0.0f);

	_decoderMemory.resize(config._decoderMemorySize);

	for (size_t i = 0; i < _decoderMemory.size(); i++) {
		std::uniform_real_distribution<float> distMemory(std::get<0>(hypernet._decoderMemoryInitRange[i]), std::get<1>(hypernet._decoderMemoryInitRange[i]));

		_decoderMemory[i] = distMemory(generator);
	}
}

void Decoder::update(const std::vector<float> &boidOutputs, class HyperNet &hypernet, float activationMultiplier) {
	// Gather inputs
	size_t inputIndex = 0;

	for (size_t i = 0; i < boidOutputs.size(); i++)
		hypernet._decoderInputBuffer[inputIndex++] = boidOutputs[i];

	for (size_t i = 0; i < _decoderMemory.size(); i++)
		hypernet._decoderInputBuffer[inputIndex++] = _decoderMemory[i];

	hypernet._decoder.process(hypernet._decoderInputBuffer, hypernet._decoderOutputBuffer, activationMultiplier);

	// Gather outputs
	size_t outputIndex = 0;

	for (size_t i = 0; i < _outputs.size(); i++)
		_outputs[i] = hypernet._decoderOutputBuffer[outputIndex++];

	for (size_t i = 0; i < _decoderMemory.size(); i++)
		_decoderMemory[i] = hypernet._decoderOutputBuffer[outputIndex++];
}