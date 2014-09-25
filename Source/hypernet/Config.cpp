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

#include <hypernet/Config.h>

using namespace hn;

Config::Config()
: _numInputGroups(2),
_numInputsPerGroup(1),
_numOutputGroups(1),
_numOutputsPerGroup(1),
_boidConnectionRadius(2.1f),
_initLinkChance(1.0f),
_boidNumOutputs(2),
_boidFiringNumHiddenLayers(1),
_boidFiringHiddenSize(26),
_boidMemorySize(4),
_boidConnectNumHiddenLayers(1),
_boidConnectHiddenSize(26),
_boidDisconnectNumHiddenLayers(1),
_boidDisconnectHiddenSize(26),
_linkResponseSize(2),
_linkNumHiddenLayers(1),
_linkHiddenSize(26),
_linkMemorySize(4),
_encoderMemorySize(3),
_decoderMemorySize(3),
_encoderNumHiddenLayers(1),
_decoderNumHiddenLayers(1),
_encoderNumPerHidden(26),
_decoderNumPerHidden(26),
_initMemoryMin(-1.0f),
_initMemoryMax(1.0f),
_initWeightMin(-0.4f),
_initWeightMax(0.4f),
_boidOutputScalar(2.0f),
_linkResponseScalar(2.0f),
_maxBoidsPerInput(3),
_maxBoidsPerOutput(3)
{}

void Config::writeToStream(std::ostream &os) const {
	// TODO: implement
	abort();
}

void Config::readFromStream(std::istream &is) {
	// TODO: implement
	abort();
}