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

#include <falcon/Falcon.h>

#include <numeric>

#include <assert.h>

#include <iostream>

using namespace falcon;

Falcon::Falcon()
: _prevQ(0.0f)
{}

float Falcon::calculateChoice(const Node &node, float minReward, float maxReward, const std::vector<float> &artInputs, const std::array<FieldParams, 3> &fieldParams, bool ignoreOutput, bool rewardChoice) const {
	if (!node._committed)
		return -999999.0f;
	
	float choice = 0.0f;

	size_t artInputIndex = 0;

	// Input
	{
		float dist = 0.0f;

		for (size_t i = 0; i < _inputs.size(); i++, artInputIndex++) {
			float delta = artInputs[artInputIndex] - node._weights[artInputIndex];
			dist += delta * delta;
		}

		//dist = std::sqrt(dist);

		choice += fieldParams[_inputField]._gamma * dist;
	}

	// Output
	if (!ignoreOutput) {
		float dist = 0.0f;

		for (size_t i = 0; i < _outputs.size(); i++, artInputIndex++) {
			float delta = artInputs[artInputIndex] - node._weights[artInputIndex];
			dist += delta * delta;
		}

		//dist = std::sqrt(dist);

		choice += fieldParams[_outputField]._gamma * dist;
	}

	// Reward
	if (rewardChoice) {
		float delta = artInputs[artInputIndex] - node._weights[artInputIndex];
		choice += fieldParams[_rewardField]._gamma * delta * delta;
	}

	return -choice;
}

bool Falcon::calculateMatch(const Node &node, const std::vector<float> &artInputs, std::array<float, 3> &vigilances) const {
	if (!node._committed)
		return true;
	
	bool match = true;
	
	size_t artInputIndex = 0;

	// Input
	{
		float dist = 0.0f;

		for (size_t i = 0; i < _inputs.size(); i++, artInputIndex++) {
			float delta = artInputs[artInputIndex] - node._weights[artInputIndex];
			dist += delta * delta;
		}

		dist = std::sqrt(dist);

		bool fieldMatch = dist < vigilances[_inputField];

		if (!fieldMatch)
			vigilances[_inputField] = dist + 0.001f;

		match = match && fieldMatch;
	}

	// Output
	{
		float dist = 0.0f;

		for (size_t i = 0; i < _outputs.size(); i++, artInputIndex++) {
			float delta = artInputs[artInputIndex] - node._weights[artInputIndex];
			dist += delta * delta;
		}

		dist = std::sqrt(dist);

		bool fieldMatch = dist < vigilances[_outputField];

		if (!fieldMatch)
			vigilances[_outputField] = dist + 0.001f;

		match = match && fieldMatch;
	}

	// Reward
	{
		float dist = 0.0f;

		for (size_t i = 0; i < 1; i++, artInputIndex++) {
			float delta = artInputs[artInputIndex] - node._weights[artInputIndex];
			dist += delta * delta;
		}

		dist = std::sqrt(dist);

		bool fieldMatch = dist < vigilances[_rewardField];

		if (!fieldMatch)
			vigilances[_rewardField] = dist + 0.001f;

		match = match && fieldMatch;
	}

	return match;
}

std::list<Falcon::Node>::iterator Falcon::findNode(const std::vector<float> &artInputs, const std::array<FieldParams, 3> &fieldParams, std::list<Node> &nodes, bool ignoreOutput, bool rewardChoice, bool runTournament, float tournamentRatio) {
	float minReward = 999999.0f;
	float maxReward = -999999.0f;
	
	for (std::list<Node>::const_iterator cit = nodes.begin(); cit != nodes.end(); cit++)
	if (cit->_committed) {
		minReward = std::min(minReward, cit->_weights[cit->_weights.size() - 2]);
		maxReward = std::max(maxReward, cit->_weights[cit->_weights.size() - 2]);
	}

	size_t tournamentSize = runTournament ? static_cast<size_t>(std::ceil(static_cast<float>(nodes.size()) * tournamentRatio)) : 1;

	std::list<std::list<Node>::iterator> tournament;

	std::list<std::list<Node>::iterator> choices;

	for (std::list<Node>::iterator it = nodes.begin(); it != nodes.end(); it++)
		choices.push_back(it);

	for (size_t i = 0; i < tournamentSize; i++) {
		std::list<std::list<Node>::iterator>::iterator maxChoiceIt = choices.begin();

		float maxChoice = calculateChoice(**maxChoiceIt, minReward, maxReward, artInputs, fieldParams, ignoreOutput, rewardChoice);

		for (std::list<std::list<Node>::iterator>::iterator it = choices.begin()++; it != choices.end(); it++) {
			float choice = calculateChoice(**it, minReward, maxReward, artInputs, fieldParams, ignoreOutput, rewardChoice);

			if (choice > maxChoice) {
				maxChoice = choice;

				maxChoiceIt = it;
			}
		}

		tournament.push_back(*maxChoiceIt);

		choices.erase(maxChoiceIt);
	}

	std::list<std::list<Node>::iterator>::iterator maxChoiceIt = tournament.begin();

	float maxChoice = (*maxChoiceIt)->_weights[(*maxChoiceIt)->_weights.size() - 2];

	for (std::list<std::list<Node>::iterator>::iterator it = tournament.begin()++; it != tournament.end(); it++) {
		float choice = (*it)->_weights[(*it)->_weights.size() - 2];

		if (choice > maxChoice) {
			maxChoice = choice;

			maxChoiceIt = it;
		}
	}

	return *maxChoiceIt;
}

void Falcon::learn(const std::vector<float> &artInputs, const std::array<FieldParams, 3> &fieldParams) {
	std::list<Node> chooseNodes = _nodes;
	std::list<Node> usedNodes;

	std::array<float, 3> vigilances;
	
	vigilances[_inputField] = fieldParams[_inputField]._baseVigilance;
	vigilances[_outputField] = fieldParams[_outputField]._baseVigilance;
	vigilances[_rewardField] = fieldParams[_rewardField]._baseVigilance;

	while (!chooseNodes.empty()) {
		std::list<Node>::iterator nodeIt = findNode(artInputs, fieldParams, chooseNodes, false, true, false, 0.0f);

		bool match = calculateMatch(*nodeIt, artInputs, vigilances);

		if (match) {
			if (!nodeIt->_committed) {
				nodeIt->_weights = artInputs;

				nodeIt->_committed = true;

				nodeIt->_eligibility = 1.0f;

				//Node newNode;

				//newNode._weights.assign(_artInputSize, 0.0f);
				//newNode._committed = false;

				//newNode._eligibility = 0.0f;

				//chooseNodes.push_back(newNode);
			}
			else {
				// Update this node
				size_t weightIndex = 0;

				for (size_t i = 0; i < _inputs.size(); i++, weightIndex++)
					nodeIt->_weights[weightIndex] = (1.0f - fieldParams[_inputField]._beta) * nodeIt->_weights[weightIndex] + fieldParams[_inputField]._beta * artInputs[weightIndex];

				for (size_t i = 0; i < _outputs.size(); i++, weightIndex++)
					nodeIt->_weights[weightIndex] = (1.0f - fieldParams[_outputField]._beta) * nodeIt->_weights[weightIndex] + fieldParams[_outputField]._beta * artInputs[weightIndex];

				for (size_t i = 0; i < 2; i++, weightIndex++)
					nodeIt->_weights[weightIndex] = (1.0f - fieldParams[_rewardField]._beta) * nodeIt->_weights[weightIndex] + fieldParams[_rewardField]._beta * artInputs[weightIndex];
			
				nodeIt->_eligibility = 1.0f;
			}

			break;
		}
		else {
			// Continue search
			//usedNodes.push_back(*nodeIt);

			//chooseNodes.erase(nodeIt);

			Node newNode;

			newNode._weights = artInputs;
			newNode._committed = true;

			newNode._eligibility = 1.0f;

			chooseNodes.push_back(newNode);

			break;
		}
	}

	assert(!chooseNodes.empty());

	// New nodes
	_nodes = usedNodes;

	for (std::list<Node>::iterator it = chooseNodes.begin(); it != chooseNodes.end(); it++)
		_nodes.push_back(*it);
}

void Falcon::create(size_t numInputs, size_t numOutputs) {
	_inputs.clear();
	_outputs.clear();
	_prevInputs.clear();
	_prevOutputs.clear();
	_nodes.clear();

	_inputs.assign(numInputs, 0.0f);
	_outputs.assign(numOutputs, 0.0f);
	_prevInputs.assign(numInputs, 0.0f);
	_prevOutputs.assign(numOutputs, 0.0f);

	_artInputSize = numInputs + numOutputs + 2;

	Node startingNode;

	startingNode._weights.assign(_artInputSize, 0.0f);
	startingNode._committed = false;

	startingNode._eligibility = 0.0f;

	_nodes.push_back(startingNode);
}

void Falcon::update(float reward, float epsilon, float gamma, float alpha, std::array<FieldParams, 3> &fieldParams, float rewardFactor, float eligibilityDecay, float tournamentRatio, std::mt19937 &generator) {
	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// Select action
	std::list<Node>::iterator it;

	if (dist01(generator) < epsilon) {
		// Random action
		for (size_t i = 0; i < _outputs.size(); i++)
			_outputs[i] = dist01(generator) * 2.0f - 1.0f;

		std::vector<float> searchVector(_artInputSize);

		size_t index = 0;

		for (size_t i = 0; i < _inputs.size(); i++)
			searchVector[index++] = _inputs[i];

		for (size_t i = 0; i < _outputs.size(); i++)
			searchVector[index++] = _outputs[i];

		searchVector[index++] = 1.0f;
		searchVector[index++] = 0.0f;

		it = findNode(searchVector, fieldParams, _nodes, false, false, false, 0.0f);
	}
	else {
		// Select action
		std::vector<float> searchVector(_artInputSize);

		size_t index = 0;

		for (size_t i = 0; i < _inputs.size(); i++)
			searchVector[index++] = _inputs[i];

		for (size_t i = 0; i < _outputs.size(); i++)
			searchVector[index++] = 0.0f;

		searchVector[index++] = 1.0f;
		searchVector[index++] = 0.0f;

		it = findNode(searchVector, fieldParams, _nodes, true, false, true, tournamentRatio);

		// Perform this action
		for (size_t i = 0; i < _outputs.size(); i++)
			_outputs[i] = it->_weights[_inputs.size() + i];
	}

	// Update Q
	float thisQ = it->_weights[_inputs.size() + _outputs.size()];

	float error = reward + gamma * thisQ - _prevQ;
	
	float newQ = _prevQ + alpha * error;

	std::cout << _nodes.size() << std::endl;

	_prevQ = thisQ;

	std::vector<float> qUpdateVector(_artInputSize);

	size_t qUpdateVectorIndex = 0;

	for (size_t i = 0; i < _prevInputs.size(); i++)
		qUpdateVector[qUpdateVectorIndex++] = _prevInputs[i];

	for (size_t i = 0; i < _prevOutputs.size(); i++)
		qUpdateVector[qUpdateVectorIndex++] = _prevOutputs[i];

	qUpdateVector[qUpdateVectorIndex++] = newQ;
	qUpdateVector[qUpdateVectorIndex++] = 0.0f;

	for (std::list<Node>::iterator it = _nodes.begin(); it != _nodes.end(); it++) {
		it->_weights[it->_weights.size() - 2] += alpha * error * it->_eligibility;
		it->_eligibility *= eligibilityDecay;
	}

	learn(qUpdateVector, fieldParams);

	_prevInputs = _inputs;
	_prevOutputs = _outputs;
}