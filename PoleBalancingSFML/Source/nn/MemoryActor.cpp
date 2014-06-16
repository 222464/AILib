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

#include <nn/MemoryActor.h>

using namespace nn;

void MemoryActor::createRandom(size_t numInputs, size_t numOutputs,
	size_t numHiddenLayers, size_t numNeuronsPerHiddenLayer,
	size_t numMemoryCells, float minWeight, float maxWeight, unsigned long seed)
{
	_actor.createRandom(numInputs + numMemoryCells, numOutputs + numMemoryCells * 4, numHiddenLayers, numNeuronsPerHiddenLayer, minWeight, maxWeight, seed);

	_memoryCells.resize(numMemoryCells);
}

void MemoryActor::step(float error) {
	for (size_t i = 0; i < _memoryCells.size(); i++)
		_actor.setInput(getNumInputs() + i, _memoryCells[i]._output);

	_actor.activateAndReinforce(error);

	size_t outputIndex = getNumOutputs();

	for (size_t i = 0; i < _memoryCells.size(); i++) {
		_memoryCells[i]._input = _actor.getOutput(outputIndex++);
		_memoryCells[i]._gateInput = _actor.getOutput(outputIndex++);
		_memoryCells[i]._gateOutput = _actor.getOutput(outputIndex++);
		_memoryCells[i]._gateForget = _actor.getOutput(outputIndex++);

		_memoryCells[i].activate(_actor._activationMultiplier, _actor._outputTraceDecay);
	}
}

void MemoryActor::writeToStream(std::ostream &stream) {

}

void MemoryActor::readFromStream(std::istream &stream) {

}