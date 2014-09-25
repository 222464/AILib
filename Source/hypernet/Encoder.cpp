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

#include <hypernet/Encoder.h>

#include <hypernet/HyperNet.h>

using namespace hn;

Encoder::Encoder(const Config &config, HyperNet &hypernet, std::mt19937 &generator) {
	_outputs.assign(config._linkResponseSize, 0.0f);

	_encoderMemory.resize(config._encoderMemorySize);

	for (size_t i = 0; i < _encoderMemory.size(); i++) {
		std::uniform_real_distribution<float> distMemory(std::get<0>(hypernet._encoderMemoryInitRange[i]), std::get<1>(hypernet._encoderMemoryInitRange[i]));

		_encoderMemory[i] = distMemory(generator);
	}
}

void Encoder::update(const std::vector<float> &input, HyperNet &hypernet, float activationMultiplier) {
	// Gather inputs
	size_t inputIndex = 0;

	for (size_t i = 0; i < input.size(); i++)
		hypernet._encoderInputBuffer[inputIndex++] = input[i];

	for (size_t i = 0; i < _encoderMemory.size(); i++)
		hypernet._encoderInputBuffer[inputIndex++] = _encoderMemory[i];

	hypernet._encoder.process(hypernet._encoderInputBuffer, hypernet._encoderOutputBuffer, activationMultiplier);

	// Gather outputs
	size_t outputIndex = 0;

	for (size_t i = 0; i < _outputs.size(); i++)
		_outputs[i] = hypernet._encoderOutputBuffer[outputIndex++];
	
	for (size_t i = 0; i < _encoderMemory.size(); i++)
		_encoderMemory[i] = hypernet._encoderOutputBuffer[outputIndex++];
}