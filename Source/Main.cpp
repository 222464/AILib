/*
AI Lib
Copyright (C) 4014 Eric Laukien

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

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Audio.hpp>

#include <Consts.h>

#include <nn/QAgent.h>
#include <nn/ActorCriticAgent.h>
#include <nn/Cacla.h>
#include <nn/MemoryActor.h>
#include <nn/RLLSTMAgent.h>
#include <nn/PSOAgent.h>
#include <nn/MultiQ.h>
#include <nn/SOMQAgent.h>

#include <lstm/LSTMActorCritic.h>

#include <deep/AutoLSTM.h>

#include <deep/FERL.h>

#include <deep/RecurrentSparseAutoencoder.h>

#include <convrl/ConvRL.h>

#include <hypernet/BayesianOptimizerTrainer.h>
#include <hypernet/EvolutionaryTrainer.h>

#include <experiments/ExperimentPoleBalancing.h>
#include <experiments/ExperimentXOR.h>
#include <experiments/ExperimentOR.h>
#include <experiments/ExperimentAND.h>

#include <hypernet/SampleField.h>
#include <hypernet/BayesianOptimizer.h>

#include <dnf/Field.h>

#include <ctrnn/GeneticAlgorithm.h>

#include <chtm/CHTMRegion.h>
#include <chtm/CHTMRL.h>

#include <nn/GeneticAlgorithm.h>

#include <falcon/Falcon.h>

#include <elman/ElmanNetwork.h>

#include <htm/Region.h>
#include <htmrl/HTMRL.h>
#include <htmrl/HTMRLDiscreteAction.h>

#include <rbf/RBFNetwork.h>

#include <plot/plot.h>

#include <deep/RBM.h>
#include <deep/DBN.h>
#include <deep/ConvNet2D.h>
#include <deep/SharpFA.h>
#include <deep/SRBMFA.h>
#include <deep/RSARL.h>

#include <raahn/AutoEncoder.h>

#include <featureExtraction/AudioFeatureMFCC.h>

#include <deep/SparseCoder.h>

#include <time.h>
#include <iostream>
#include <random>
#include <array>
#include <fstream>
#include <sstream>

#include <text/Word2SDR.h>

#include <dirent.h>

/*int main() {
	std::mt19937 generator(time(nullptr));
	
	hn::FunctionApproximator approx;
	approx.createRandom(2, 1, 1, 4, -0.2f, 0.2f, generator);

	float inputs[4][2] {
		{ 0.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		0.0f,
		1.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;
	float totalReward = 0.0f;

	std::vector<float> inputBuffer(2);
	std::vector<float> outputBuffer(1);

	for (size_t i = 0; i < 10000; i++) {
		float newReward = 1.0f;

		for (size_t j = 0; j < 4; j++) {
			inputBuffer[0] = (inputs[j][0]);
			inputBuffer[1] = (inputs[j][1]);

			std::vector<std::vector<float>> layerOutputs;

			approx.process(inputBuffer, layerOutputs, 1.0f);

			newReward -= std::abs(outputs[j] - layerOutputs.back()[0]) * 0.25f;

			approx.backpropagate(inputBuffer, layerOutputs, std::vector<float>(1, outputs[j]), 0.1f);
		}

		prevReward = reward;
		reward = newReward;

		totalReward += newReward;
	}

	for (size_t j = 0; j < 4; j++) {
		inputBuffer[0] = (inputs[j][0]);
		inputBuffer[1] = (inputs[j][1]);

		approx.process(inputBuffer, outputBuffer, 1.0f);

		std::cout << outputBuffer[0] << std::endl;
	}

	// Initialize input to something random
	inputBuffer[0] = 0.2f;
	inputBuffer[1] = 0.6f;

	float target = 1.0f;

	for (int i = 0; i < 1000; i++) {
		std::vector<std::vector<float>> layerOutputs;

		approx.process(inputBuffer, layerOutputs, 1.0f);

		std::vector<float> inputErrors;

		approx.getInputError(inputBuffer, layerOutputs, std::vector<float>(1, target), inputErrors);

		for (int j = 0; j < inputBuffer.size(); j++) {
			inputBuffer[j] = std::min(1.0f, std::max(0.0f, inputBuffer[j] + 0.1f * inputErrors[j]));

			std::cout << inputBuffer[j] << " ";
		}
		
		std::cout << layerOutputs.back().front() << std::endl;

		system("pause");
	}

	system("pause");
}*/

	/*hn::EvolutionaryTrainer trainer;
	hn::Config config;
	
	trainer.create(config, 32, generator, 6.0f);

	trainer.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentXOR()));
	trainer.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentOR()));
	trainer.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentAND()));

	for (size_t g = 0; g < 10000; g++) {
		std::cout << "Evaluating generation " << (g + 1) << "." << std::endl;

		trainer.evaluate(config, generator);

		std::cout << "Reproducing generation " << (g + 1) << "." << std::endl;

		trainer.reproduce(config, generator);

		std::cout << "Saving best to \"hypernet1.txt\"" << std::endl;

		std::ofstream toFile("hypernet1.txt");

		trainer.writeBestToStream(toFile);

		toFile.close();

		std::cout << "Generation completed." << std::endl;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			break;
	}

	return 0;
}*/

/*float testFunc(float x) {
	return 0.2f * x * x * x * x + 0.5f * x * x * x - 1.2f * x * x + 0.2f * x + 0.3f;
}*/

/*int main() {*/
	/*sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	hn::SampleField sampleField;
	sampleField.create(2, 3);
	sampleField._kernel = std::bind(hn::kernelSquaredExponential, std::placeholders::_1, std::placeholders::_2, 128.0f);

	std::mt19937 generator(time(nullptr));

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	// Add some samples
	for (size_t i = 0; i < 100; i++) {
		hn::SampleField::Sample s;

		s._x.resize(2);
		s._x[0] = dist01(generator);
		s._x[1] = dist01(generator);
		s._y.resize(3);
		s._y[0] = dist01(generator);
		s._y[1] = dist01(generator);
		s._y[2] = dist01(generator);

		sampleField.addSample(s);
	}
	
	sf::Image image;

	image.create(800, 600);

	for (unsigned int x = 0; x < image.getSize().x; x++)
	for (unsigned int y = 0; y < image.getSize().y; y++) {
		std::vector<float> sx(2);
		sx[0] = static_cast<float>(x) / static_cast<float>(image.getSize().x);
		sx[1] = static_cast<float>(y) / static_cast<float>(image.getSize().y);

		std::vector<float> sy = sampleField.getYAtX(sx);

		sf::Color c;

		c.r = sy[0] * 255.0f;
		c.g = sy[1] * 255.0f;
		c.b = sy[2] * 255.0f;

		image.setPixel(x, y, c);
	}

	sf::Texture texture;
	texture.loadFromImage(image);

	// ----------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// ---------------------------- Rendering ----------------------------

		sf::Sprite s;
		s.setTexture(texture);

		window.draw(s);

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	return 0;*/

	/*std::mt19937 generator(time(nullptr));
	hn::BayesianOptimizer bo;

	bo.create(1, std::vector<float>(1, -3.0f), std::vector<float>(1, 1.0f));

	for (size_t i = 0; i < 6; i++) {
		bo.generateNewVariables(generator);

		bo.update(testFunc(bo.getCurrentVariables()[0]));
	}

	std::cout << bo.getCurrentVariables()[0] << std::endl;

	system("pause");*/

	/*std::mt19937 generator(time(nullptr));

	hn::Config config;
	hn::BayesianOptimizerTrainer bot;

	bot.create(config, -2.0f, 2.0f, generator);

	bot.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentXOR()));
	bot.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentOR()));
	bot.addExperiment(std::shared_ptr<hn::Experiment>(new ExperimentAND()));

	for (size_t g = 0; g < 1000; g++) {
		bot.evaluate(config, generator);

		std::cout << "Current fitness: " << bot.getCurrentFitness() << std::endl;

		bot.update(config, generator);
	}

	return 0;
}*/

/*int main() {
	float inputs[4][2] {
		{ -10.0f, -10.0f },
		{ -10.0f, 10.0f },
		{ 10.0f, -10.0f },
		{ 10.0f, 10.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	hn::Config config;
	config._numInputGroups = 2;
	config._numInputsPerGroup = 1;
	config._numOutputGroups = 1;
	config._numOutputsPerGroup = 1;

	std::mt19937 generator(time(nullptr));
	hn::HyperNet hypernet;
	hypernet.readFromStream(std::ifstream("hypernet1.txt"));

	std::vector<int> dims(2);
	dims[0] = 12;
	dims[1] = 6;
	hypernet.generateFeedForward(config, 1, 4, generator);

	float reward = 0.0f;
	float prevReward = 0.0f;
	float totalReward = 0.0f;

	for (size_t i = 0; i < 400; i++) {
		float newReward = 1.0f;

		for (size_t j = 0; j < 4; j++) {
			hypernet.setInput(0, inputs[j][0]);
			hypernet.setInput(1, inputs[j][1]);

			hypernet.step(config, (reward - prevReward) * 50.0f, generator, 6, 6.0f);

			newReward -= std::abs(outputs[j] - hypernet.getOutput(0)) * 0.25f;
		}

		prevReward = reward;
		reward = newReward;

		totalReward += newReward;
	}

	for (size_t j = 0; j < 4; j++) {
		hypernet.setInput(0, inputs[j][0]);
		hypernet.setInput(1, inputs[j][1]);

		hypernet.step(config, (reward - prevReward) * 50.0f, generator, 6, 6.0f);

		std::cout << hypernet.getOutput(0) << std::endl;
	}

	std::cout << "totalReward: " << totalReward << std::endl;

	system("pause");

	return 0;
}*/

/*
int main() {
	std::mt19937 generator(time(nullptr));

	raahn::AutoEncoder ae;

	ae.create(10, 3, -0.3f, 0.3f, generator);

	std::vector<float> inputs(10, 0.0f);
	std::vector<float> outputs(4, 0.0f);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	for (size_t j = 0; j < 10; j++) {
		inputs[j] = dist01(generator);
	}

	for (size_t i = 0; i < 1000; i++) {
		

		ae.update(inputs, outputs, 0.1f);

		float error = 0.0f;

		for (size_t j = 0; j < 10; j++)
			error += ae.getInputErrorBuffer()[j];

		std::cout << "Total Error: " << error << std::endl;

		
	}

	system("pause");

	return 0;
}*/

float evaluatePoleBalancing(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	float dt = 0.03f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float totalFitness = 0.0f;

	for (size_t i = 0; i < 1500; i++) {
		//std::cout << "Step " << i << std::endl;

		// Update fitness
		prevFitness = fitness;

		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		//fitness = fitness - std::abs(poleAngleVel * 40.0f);

		totalFitness += fitness * 0.01f;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		net.setInput(0, cartX * 0.25f);
		net.setInput(1, cartVelX);
		net.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		net.setInput(3, poleAngleVel);

		net.setInput(4, dFitness * 0.1f);

		for (size_t s = 0; s < 18; s++)
			net.step(1.0f, dFitness * 0.1f, 0.1f, generator);

		float dir = std::min(1.0f, std::max(-1.0f, 2.0f * net.getOutput(0) - 1.0f));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 4000.0f * agent.getOutput(0);

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed)
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(PI)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(PI) * 2.0f;
	}

	//std::cout << "Pole balancing experiment finished with total fitness of " << totalFitness << "." << std::endl;

	return totalFitness;
}

float evaluateAND(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		0.0f,
		0.0f,
		1.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			net.setInput(0, inputs[j][0]);
			net.setInput(1, inputs[j][1]);
			net.setInput(2, inputs[j][0]);
			net.setInput(3, inputs[j][1]);
			net.setInput(4, (reward - prevReward) * 0.1f);

			for(size_t s = 0; s < 18; s++)
				net.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - net.getOutput(0)), 1.0f) * 0.25f;

			values[j] = net.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.1f;
	}

	//std::cout << "Finished AND experiment with total reward of " << (reward - initReward) * std::min(1.0f, std::max(0.0f, (reward * reward))) << "." << std::endl;

	return totalReward;
}

float evaluateOR(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		1.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			net.setInput(0, inputs[j][0]);
			net.setInput(1, inputs[j][1]);
			net.setInput(2, inputs[j][0]);
			net.setInput(3, inputs[j][1]);
			net.setInput(4, (reward - prevReward) * 0.1f);

			for (size_t s = 0; s < 18; s++)
				net.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - net.getOutput(0)), 1.0f) * 0.25f;

			values[j] = net.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.5f;
	}

	//std::cout << "Finished AND experiment with total reward of " << (reward - initReward) * std::min(1.0f, std::max(0.0f, (reward * reward))) << "." << std::endl;

	return totalReward;
}

float evaluateXOR(ctrnn::CTRNN &net, std::mt19937 &generator) {
	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			net.setInput(0, inputs[j][0]);
			net.setInput(1, inputs[j][1]);
			net.setInput(2, inputs[j][0]);
			net.setInput(3, inputs[j][1]);
			net.setInput(4, (reward - prevReward) * 0.1f);

			for (size_t s = 0; s < 18; s++)
				net.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - net.getOutput(0)), 1.0f) * 0.25f;

			values[j] = net.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.5f;
	}

	//std::cout << "Finished AND experiment with total reward of " << (reward - initReward) * std::min(1.0f, std::max(0.0f, (reward * reward))) << "." << std::endl;

	return totalReward;
}

/*int main() {
	std::mt19937 generator(time(nullptr));

	ctrnn::GeneticAlgorithm ga;

	ga.create(40, 5, 1, 14, -0.5f, 0.5f, 1.01f, 2.0f, 0.05f, 1.0f, generator);

	for (size_t g = 0; g < 550; g++) {
		float maxFitness = -99999.0f;

		for (size_t i = 0; i < ga.getPopulationSize(); i++) {
			//ctrnn::CTRNN memberPoleBalancing = ga.getPopulationMember(i);
			ctrnn::CTRNN memberAND = ga.getPopulationMember(i);
			ctrnn::CTRNN memberOR = ga.getPopulationMember(i);
			ctrnn::CTRNN memberXOR = ga.getPopulationMember(i);

			ga.setFitness(i, evaluateAND(memberAND, generator) + evaluateOR(memberOR, generator) + evaluateXOR(memberXOR, generator));

			if (ga.getFitness(i) > maxFitness)
				maxFitness = ga.getFitness(i);
		}

		ga.generation(0.3f, 0.05f, 0.2f, 0.3f, 0.05f, 0.2f, 0.3f, 0.05f, 0.2f, 3.0f, 4, generator);

		std::cout << "Finished generation " << g << ". Peak fitness: " << maxFitness << std::endl;
	}

	float maxFitness = -99999.0f;
	size_t maxFitnessIndex = 0;

	for (size_t i = 0; i < ga.getPopulationSize(); i++) {
		//ctrnn::CTRNN memberPoleBalancing = ga.getPopulationMember(i);
		ctrnn::CTRNN memberAND = ga.getPopulationMember(i);
		ctrnn::CTRNN memberOR = ga.getPopulationMember(i);
		ctrnn::CTRNN memberXOR = ga.getPopulationMember(i);

		float f = evaluateAND(memberAND, generator) + evaluateOR(memberOR, generator) + evaluateXOR(memberXOR, generator);

		if (f > maxFitness) {
			maxFitness = f;
			maxFitnessIndex = i;
		}
	}

	ctrnn::CTRNN best = ga.getPopulationMember(maxFitnessIndex);

	best.clear();

	float inputs[4][2] {
		{ -1.0f, -1.0f },
		{ -1.0f, 1.0f },
		{ 1.0f, -1.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	std::vector<float> values(4);

	for (size_t i = 0; i < 100; i++) {
		float newReward = 1.0f;

		float average = 0.0f;

		for (size_t j = 0; j < 4; j++) {
			best.setInput(0, inputs[j][0]);
			best.setInput(1, inputs[j][1]);
			best.setInput(2, inputs[j][0]);
			best.setInput(3, inputs[j][1]);
			best.setInput(4, (reward - prevReward) * 0.1f);

			for (size_t s = 0; s < 16; s++)
				best.step(1.0f, (reward - prevReward) * 0.1f, 0.1f, generator);

			newReward -= std::pow(std::abs(outputs[j] - best.getOutput(0)), 2.0f) * 0.25f;

			if (i == 49)
				std::cout << best.getOutput(0) << std::endl;

			values[j] = best.getOutput(0);
			average += values[j];
		}

		average *= 0.25f;

		//for (size_t j = 0; j < 4; j++) {
		//	newReward -= 0.05f / (0.125f + 8.0f * std::abs(average - values[j]));
		//}

		prevReward = reward;
		reward = newReward;

		if (i == 0)
			initReward = std::min(1.0f, std::max(0.0f, reward));

		totalReward += reward * 0.5f;
	}

	best.clear();

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	
	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		//fitness = fitness - std::abs(poleAngleVel * 40.0f);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - lowPassFitness;

		float error = dFitness * 10.0f;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		best.setInput(0, cartX * 0.25f);
		best.setInput(1, cartVelX);
		best.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		best.setInput(3, poleAngleVel);

		best.setInput(4, dFitness * 10.0f);

		for (size_t s = 0; s < 16; s++)
			best.step(1.0f, dFitness, 0.1f, generator);

		lowPassFitness += (fitness - lowPassFitness) * 0.2f;

		//std::cout << dFitness * 0.5f - 0.001f << std::endl;

		float dir = std::min(1.0f, std::max(-1.0f, 2.0f * best.getOutput(0) - 1.0f));

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		float agentForce = 4000.0f * dir;
		//float agentForce = 4000.0f * agent.getOutput(0);

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(PI)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(PI) * 2.0f;

		// ---------------------------- Rendering ----------------------------

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + pixelsPerMeter * cartX, static_cast<float>(window.getSize().y) * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI) + 180.0f);

		window.draw(poleSprite);

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow window;
	window.create(sf::VideoMode(512, 512), "AILib", sf::Style::Default);

	dnf::Field field;
	field.createRandom(2, 32, 2, 0.75f, 0.0f, 1.0f, generator);

	float size = std::min(window.getSize().x / static_cast<float>(field.getSize()), window.getSize().y / static_cast<float>(field.getSize()));

	sf::Vector2f boxSize(size, size);

	// -------------------------- Simulation Loop -------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float switchPatternTimer = 0.0f;
	float switchPatternTime = 280.0f;

	int pattern = 0;

	float inputPattern[4][2] = {
		{ 0.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f }
	};

	float outputPattern[4] = {
		0.0f,
		0.0f,
		0.0f,
		0.0f
	};

	float reward = 0.0f;
	float prevReward = 0.0f;

	float averageReward = 0.0f;

	float fieldAverage = 0.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		std::vector<float> input(4);

		input[0] = inputPattern[pattern][0];
		input[1] = inputPattern[pattern][1];

		// -------------------------------------------------------------------

		window.clear();

		for (int i = 0; i < 1; i++)
		field.step(input, 0.1f, 0.2f, 0.3f, (reward - prevReward) * 0.001f, 0.05f, 0.001f, 0.001f, 0.002f, 0.5f);

		// Draw grid
		int lIndex = 0;

		for (int x = 0; x < field.getSize(); x++)
		for (int y = 0; y < field.getSize(); y++) {
			sf::RectangleShape rectShape;

			rectShape.setPosition(x * boxSize.x, y * boxSize.y);
			rectShape.setSize(boxSize);

			float positivePotentialColor = field.getOutputE(x, y);
			float negativePotentialColor = field.getOutputE(x, y);

			rectShape.setFillColor(sf::Color(positivePotentialColor * 255.0f, negativePotentialColor * 255.0f, 0));

			window.draw(rectShape);

			lIndex++;
		}

		switchPatternTimer += 1.0f;

		if (switchPatternTimer > switchPatternTime) {
			prevReward = reward;

			reward = -std::pow(std::abs(outputPattern[pattern] - std::min(1.0f, std::max(0.0f, field.getOutputE(16, 16)))), 2.0f);

			averageReward += (reward - averageReward) * 0.1f;

			std::cout << field.getOutputE(16, 16) << "    " << averageReward << "     " << reward << std::endl;

			switchPatternTimer -= switchPatternTime;

			pattern = (pattern + 1) % 4;

			fieldAverage = 0.0f;
		}

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	float inputs[4][2] {
		{ 0.0f, 0.0f },
		{ 0.0f, 1.0f },
		{ 1.0f, 0.0f },
		{ 1.0f, 1.0f }
	};

	float outputs[4] {
		0.0f,
		1.0f,
		1.0f,
		0.0f
	};

	rbf::RBFNetwork fa;

	fa.createRandom(2, 32, 1, -0.5f, 0.5f, 0.01f, 2.0f, -0.1f, 0.1f, generator);

	std::vector<float> in(2);
	std::vector<float> out(1);

	for (int i = 0; i < 50; i++) {
		//fa.clearGradient();

		for (int k = 0; k < 4; k++) {
			in[0] = inputs[k][0];
			in[1] = inputs[k][1];

			//fa.process(in, out);

			//fa.accumulateGradient(in, std::vector<float>(1, outputs[k]));

			fa.getOutput(in, out);

			fa.learnFeatures(in, 0.2f, 0.2f, 0.5f);

			fa.update(in, out, std::vector<float>(1, outputs[k]), 0.1f);
		}

		//fa.scaleGradient(0.25f);

		//fa.moveAlongGradientRMS(0.1f, 0.1f, 0.8f, 0.0f, 0.01f);
	}

	for (int k = 0; k < 4; k++) {
		in[0] = inputs[k][0];
		in[1] = inputs[k][1];

		//fa.process(in, out);

		fa.getOutput(in, out);

		std::cout << out[0] << std::endl;
	}

	system("pause");

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI) * 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	//deep::AutoLSTM alstm;

	//elman::ElmanNetwork en;

	//en.createRandom(4, 1, 10, -0.5f, 0.5f, generator);

	//lstm::LSTMActorCritic lstmAC;

	//lstmAC.createRandom(4, 1, 1, 24, 2, 1, 1, 24, 2, 1, -0.5f, 0.5f, generator);

	htmrl::HTMRLDiscreteAction htmRL;
	std::vector<htmrl::HTMRLDiscreteAction::RegionDesc> regionDescs(1);
	regionDescs[0]._regionWidth = 32;
	regionDescs[0]._regionHeight = 16;

	std::vector<float> condensed;

	htmRL.createRandom(2, 1, 32, 32, 2, 2, 3, 1, 32, 0.1f, regionDescs, generator);

	//falcon::Falcon fal;
	//fal.create(4, 1);

	//deep::FERL ferl;
	//ferl.createRandom(4, 1, 24, 0.1f, generator);

	//nn::SOMQAgent sqa;

	//sqa.createRandom(4, 1, 2, 60, nn::BrownianPerturbation(), -0.5f, 0.5f, generator);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Image image;

	image.create(htmRL.getCondenseBufferWidth(), htmRL.getCondenseBufferHeight());

	sf::RenderTexture rt;

	rt.create(800, 600);

	sf::plot::Plot plot;

	plot.setSize(sf::Vector2f(rt.getSize().x, rt.getSize().y));
	plot.setTitle("Avg. Reward");
	plot.setFont("Resources/arial.ttf");
	plot.setXLabel("Time (seconds)");
	plot.setYLabel("Avg. Reward");
	plot.setBackgroundColor(sf::Color::White);
	plot.setTitleColor(sf::Color::Black);
	plot.setPosition(sf::Vector2f(0.0f, 0.0f));

	sf::plot::Curve &cAverage = plot.createCurve("Avg. Reward", sf::Color::Red);

	cAverage.setFill(false);

	plot.prepare();

	float plotMin = 99999.0f;
	float plotMax = -99999.0f;

	rt.draw(plot);
	rt.display();

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI) * 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI) * 0.5f - (static_cast<float>(PI) * 2.0f - poleAngle));

		fitness += static_cast<float>(PI) * 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.2f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		//agent.setInput(0, 0, cartX * 0.5f);
		//agent.setInput(0, 1, cartVelX);
		//agent.setInput(1, 0, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI) / PI * 2.0f - 1.0f));
		//agent.setInput(1, 1, poleAngleVel * 0.1f);

		//agent.step(dFitness * 6.0f, 0.97f, 0.5f, 0.1f, 0.001f, regionDesc, 0.2f, 0.3f, 0.05f, generator);

		//lstmAC.setInput(0, cartX * 0.25f);
		//lstmAC.setInput(1, cartVelX);
		//lstmAC.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		//lstmAC.setInput(3, poleAngleVel);

		//lstmAC.step(dFitness * 10.0f, 0.2f, 0.01f, 0.02f, 0.4f, 0.0001f, 0.97f, 0.5f, 0.5f, 0.05f, 0.2f, 0.2f, 0.99f, generator);

		//fal.update(fitness, 0.1f, 0.9f, 0.2f, fieldParams, 1.5f, 0.8f, 0.07f, generator);

		//sqa.step(fitness, 0.3f, 0.96f, 0.5f, 0.07f, dt, generator);

		std::vector<float> state(4);

		state[0] = std::min(1.0f, std::max(-1.0f, cartX * 0.333f));
		state[1] = std::min(1.0f, std::max(-1.0f, cartVelX * 0.25f));
		state[2] = std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)) / (2.0f * static_cast<float>(PI)) * 2.0f - 1.0f;
		state[3] = std::min(1.0f, std::max(-1.0f, poleAngleVel * 0.25f));

		std::vector<float> output;

		htmRL.setInput(0, 0, 0, state[0]);
		htmRL.setInput(0, 0, 1, state[1]);
		htmRL.setInput(1, 0, 0, state[2]);
		htmRL.setInput(1, 0, 1, state[3]);

		int action;

		action = htmRL.step(reward, 0.5f, 0.05f, 0.005f, 0.2f, 0.997f, 0.99f, 1.0f, 0.15f, 0.0f, 0.0f, 0.0f, 0.05f, generator, condensed);

		output.resize(1);
		output[0] = action - 1;

		//ferl.step(state, output, reward, 0.0001f, 0.97f, 0.8f, 6.0f, 8, 8, 0.05f, 0.05f, 0.1f, generator);

		lowPassFitness += (fitness - lowPassFitness) * 0.2f;

		float agentForce = 0.0f;

		//if (!trainMode) {
		//float dir = std::min(1.0f, std::max(-1.0f, agent.getOutput(0)));

		float dir = output[0];

			//dir = 1.4f * (dir * 2.0f - 1.0f);

			agentForce = 4000.0f * dir;
		//}
		//float agentForce = 4000.0f * agent.getOutput(0);

			prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI) * 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		//if (trainMode) {
		//	en.calculateGradient(std::vector<float>(1, force / 4000.0f));

		//	en.moveAlongGradient(0.01f);
		//}

		//en.updateContext();

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(PI)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(PI) * 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}
			
			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI) + 180.0f);

		window.draw(poleSprite);

		if (plotUpdateTimer >= 1.0f) {
			plotUpdateTimer -= 1.0f;

			plotMin = std::min<float>(avgReward, plotMin);
			plotMax = std::max<float>(avgReward, plotMax);

			cAverage.addValue(avgReward);

			plot.prepare();

			cAverage.prepare(sf::Vector2f(0.0f, totalTime), sf::Vector2f(plotMin, plotMax));

			rt.draw(plot);

			rt.display();
		}

		sf::Sprite plotSprite;

		plotSprite.setPosition(800.0f, 0.0f);

		plotSprite.setTexture(rt.getTexture());

		window.draw(plotSprite);

		if (trainMode) {
			sf::Text text;

			text.setFont(font);

			text.setString("Train");

			text.setPosition(10.0f, 10.0f);

			text.setColor(sf::Color::Blue);

			window.draw(text);
		}

		for (size_t i = 0; i < htmRL.getCondenseBufferWidth() * htmRL.getCondenseBufferHeight(); i++) {
			int x = i % htmRL.getCondenseBufferWidth();
			int y = i / htmRL.getCondenseBufferWidth();

			sf::Color c;
			
			c.r = condensed[i] * 255.0f;
			c.g = 0;
			c.b = 0;

			image.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setPosition(800.0f - 256.0f, 0.0f);

		s.setScale(256.0f / htmRL.getCondenseBufferWidth(), 256.0f / htmRL.getCondenseBufferWidth());

		s.setTexture(t);

		window.draw(s);
		
		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	sf::Texture inputCartTexture;
	sf::Texture inputPoleTexture;

	inputCartTexture.loadFromFile("Resources/inputCart.png");
	inputPoleTexture.loadFromFile("Resources/inputPole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	sf::Sprite inputCartSprite;
	sf::Sprite inputPoleSprite;

	inputCartSprite.setTexture(inputCartTexture);
	inputPoleSprite.setTexture(inputPoleTexture);

	inputCartSprite.setOrigin(sf::Vector2f(static_cast<float>(inputCartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputCartSprite.getTexture()->getSize().y)));
	inputPoleSprite.setOrigin(sf::Vector2f(static_cast<float>(inputPoleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputPoleSprite.getTexture()->getSize().y)));


	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float inputPixelsPerMeter = 8.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI)* 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	//deep::AutoLSTM alstm;

	//elman::ElmanNetwork en;

	//en.createRandom(4, 1, 10, -0.5f, 0.5f, generator);

	//lstm::LSTMActorCritic lstmAC;

	//lstmAC.createRandom(4, 1, 1, 24, 2, 1, 1, 24, 2, 1, -0.5f, 0.5f, generator);

	htmrl::HTMRLDiscreteAction htmRL;
	std::vector<htmrl::HTMRLDiscreteAction::RegionDesc> regionDescs(3);
	regionDescs[0]._connectionRadius = 1;
	regionDescs[0]._initInhibitionRadius = 1.0f;
	regionDescs[0]._desiredLocalActivity = 1;
	regionDescs[0]._learningRadius = 1;
	regionDescs[0]._regionWidth = 128;
	regionDescs[0]._regionHeight = 128;

	regionDescs[1]._connectionRadius = 2;
	regionDescs[1]._initInhibitionRadius = 2.0f;
	regionDescs[1]._desiredLocalActivity = 2;
	regionDescs[1]._learningRadius = 2;
	regionDescs[1]._regionWidth = 64;
	regionDescs[1]._regionHeight = 64;

	regionDescs[2]._connectionRadius = 2;
	regionDescs[2]._initInhibitionRadius = 2.0f;
	regionDescs[2]._desiredLocalActivity = 2;
	regionDescs[2]._learningRadius = 2;
	regionDescs[2]._regionWidth = 32;
	regionDescs[2]._regionHeight = 32;

	std::vector<float> condensed;

	htmRL.createRandom(32, 32, 4, 4, 2, 2, 3, 1, 32, 0.1f, regionDescs, generator);

	//falcon::Falcon fal;
	//fal.create(4, 1);

	//deep::FERL ferl;
	//ferl.createRandom(4, 1, 24, 0.1f, generator);

	//nn::SOMQAgent sqa;

	//sqa.createRandom(4, 1, 2, 60, nn::BrownianPerturbation(), -0.5f, 0.5f, generator);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Image image;

	image.create(htmRL.getCondenseBufferWidth(), htmRL.getCondenseBufferHeight());

	sf::RenderTexture inputRT;

	inputRT.create(64, 32);

	sf::RenderTexture rt;

	rt.create(800, 600);

	sf::plot::Plot plot;

	plot.setSize(sf::Vector2f(rt.getSize().x, rt.getSize().y));
	plot.setTitle("Avg. Reward");
	plot.setFont("Resources/arial.ttf");
	plot.setXLabel("Time (seconds)");
	plot.setYLabel("Avg. Reward");
	plot.setBackgroundColor(sf::Color::White);
	plot.setTitleColor(sf::Color::Black);
	plot.setPosition(sf::Vector2f(0.0f, 0.0f));

	sf::plot::Curve &cAverage = plot.createCurve("Avg. Reward", sf::Color::Red);

	cAverage.setFill(false);

	plot.prepare();

	float plotMin = 99999.0f;
	float plotMax = -99999.0f;

	rt.draw(plot);
	rt.display();

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI)* 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI)* 0.5f - (static_cast<float>(PI)* 2.0f - poleAngle));

		fitness += static_cast<float>(PI) * 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.2f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		//agent.reinforceArp(std::min(1.0f, std::max(-1.0f, error)) * 0.5f + 0.5f, 0.1f, 0.05f);

		//agent.setInput(0, 0, cartX * 0.5f);
		//agent.setInput(0, 1, cartVelX);
		//agent.setInput(1, 0, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI) / PI * 2.0f - 1.0f));
		//agent.setInput(1, 1, poleAngleVel * 0.1f);

		//agent.step(dFitness * 6.0f, 0.97f, 0.5f, 0.1f, 0.001f, regionDesc, 0.2f, 0.3f, 0.05f, generator);

		//lstmAC.setInput(0, cartX * 0.25f);
		//lstmAC.setInput(1, cartVelX);
		//lstmAC.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)));
		//lstmAC.setInput(3, poleAngleVel);

		//lstmAC.step(dFitness * 10.0f, 0.2f, 0.01f, 0.02f, 0.4f, 0.0001f, 0.97f, 0.5f, 0.5f, 0.05f, 0.2f, 0.2f, 0.99f, generator);

		//fal.update(fitness, 0.1f, 0.9f, 0.2f, fieldParams, 1.5f, 0.8f, 0.07f, generator);

		//sqa.step(fitness, 0.3f, 0.96f, 0.5f, 0.07f, dt, generator);

		//std::vector<float> state(4);

		//state[0] = std::min(1.0f, std::max(-1.0f, cartX * 0.333f));
		//state[1] = std::min(1.0f, std::max(-1.0f, cartVelX * 0.25f));
		//state[2] = std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)) / (2.0f * static_cast<float>(PI)) * 2.0f - 1.0f;
		//state[3] = std::min(1.0f, std::max(-1.0f, poleAngleVel * 0.25f));

		//htmRL.setInput(0, 0, 0, state[0]);
		//htmRL.setInput(0, 0, 1, state[1]);
		//htmRL.setInput(1, 0, 0, state[2]);
		//htmRL.setInput(1, 0, 1, state[3]);

		sf::Image img = inputRT.getTexture().copyToImage();

		for (int i = 0; i < htmRL.getInputWidth() * htmRL.getInputHeight(); i++) {
			int x = i % htmRL.getInputWidth();
			int y = i / htmRL.getInputWidth();

			int xl = x * 2;
			int yl = y;

			htmRL.setInput(x, y, 0, img.getPixel(xl, yl).r / 255.0f * 2.0f - 1.0f);
			htmRL.setInput(x, y, 1, img.getPixel(xl + 1, yl).r / 255.0f * 2.0f - 1.0f);
		}

		std::vector<float> output;

		int action;

		action = htmRL.step(reward, 0.5f, 0.05f, 0.005f, 0.2f, 0.994f, 0.99f, 1.0f, 0.15f, 0.0f, 0.0f, 0.0f, 0.05f, generator, condensed);

		output.resize(1);
		output[0] = action - 1;

		//ferl.step(state, output, reward, 0.0001f, 0.97f, 0.8f, 6.0f, 8, 8, 0.05f, 0.05f, 0.1f, generator);

		lowPassFitness += (fitness - lowPassFitness) * 0.2f;

		float agentForce = 0.0f;

		//if (!trainMode) {
		//float dir = std::min(1.0f, std::max(-1.0f, agent.getOutput(0)));

		float dir = output[0];

		//dir = 1.4f * (dir * 2.0f - 1.0f);

		agentForce = 4000.0f * dir;
		//}
		//float agentForce = 4000.0f * agent.getOutput(0);

		prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		//if (trainMode) {
		//	en.calculateGradient(std::vector<float>(1, force / 4000.0f));

		//	en.moveAlongGradient(0.01f);
		//}

		//en.updateContext();

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(PI)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(PI)* 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}

			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		// Render to input buffer
		inputRT.clear();

		inputCartSprite.setPosition(sf::Vector2f(inputRT.getSize().x * 0.5f + inputPixelsPerMeter * cartX, inputRT.getSize().y * 0.5f + 4.0f));

		inputRT.draw(inputCartSprite);

		inputPoleSprite.setPosition(inputCartSprite.getPosition() + sf::Vector2f(0.0f, -4.0f));
		inputPoleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI) + 180.0f);

		inputRT.draw(inputPoleSprite);

		inputRT.display();

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI)+180.0f);

		window.draw(poleSprite);

		if (plotUpdateTimer >= 1.0f) {
			plotUpdateTimer -= 1.0f;

			plotMin = std::min<float>(avgReward, plotMin);
			plotMax = std::max<float>(avgReward, plotMax);

			cAverage.addValue(avgReward);

			plot.prepare();

			cAverage.prepare(sf::Vector2f(0.0f, totalTime), sf::Vector2f(plotMin, plotMax));

			rt.draw(plot);

			rt.display();
		}

		sf::Sprite plotSprite;

		plotSprite.setPosition(800.0f, 0.0f);

		plotSprite.setTexture(rt.getTexture());

		window.draw(plotSprite);

		if (trainMode) {
			sf::Text text;

			text.setFont(font);

			text.setString("Train");

			text.setPosition(10.0f, 10.0f);

			text.setColor(sf::Color::Blue);

			window.draw(text);
		}

		for (size_t i = 0; i < htmRL.getCondenseBufferWidth() * htmRL.getCondenseBufferHeight(); i++) {
			int x = i % htmRL.getCondenseBufferWidth();
			int y = i / htmRL.getCondenseBufferWidth();

			sf::Color c;

			c.r = condensed[i] * 255.0f;
			c.g = 0;
			c.b = 0;

			image.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setPosition(800.0f - 256.0f, 0.0f);

		s.setScale(256.0f / htmRL.getCondenseBufferWidth(), 256.0f / htmRL.getCondenseBufferWidth());

		s.setTexture(t);

		window.draw(s);

		sf::Sprite inputSprite;

		inputSprite.setTexture(inputRT.getTexture());

		inputSprite.setPosition(0, 0);
		inputSprite.setScale(4.0f, 4.0f);

		window.draw(inputSprite);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);
}*/

/*int main() {
	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "Mountain Car");

	//window.setVerticalSyncEnabled(true);

	window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;

	cartTexture.loadFromFile("Resources/car.png");

	// --------------------------------------------------------------------

	sf::Sprite cartSprite;

	cartSprite.setTexture(cartTexture);

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y) * 0.5f));
	cartSprite.setScale(sf::Vector2f(0.25f, 0.25f));

	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;

	float velocity = 0.0f;
	float position = -0.5f;

	// ------------------------------- AI ---------------------------------

	std::mt19937 generator(time(nullptr));

	chtm::CHTMRL htm;

	htm.createRandom(2, 2, 16, 16, 3, 1, 3, -0.75f, 0.75f, 0.01f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, generator);

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	prevFitness = fitness;
	fitness = 0.0f;

	std::vector<sf::Vertex> hills;

	for (float p = -3.141f; p < 3.141f;) {
		sf::Vertex v1;
		v1.position = sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + p * pixelsPerMeter, static_cast<float>(window.getSize().y) * 0.5f + std::sin(p * 3.0f) * pixelsPerMeter * -0.333f);
		v1.color = sf::Color::Red;

		p += 0.1f;

		sf::Vertex v2;
		v2.position = sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + p * pixelsPerMeter, static_cast<float>(window.getSize().y) * 0.5f + std::sin(p * 3.0f) * pixelsPerMeter * -0.333f);
		v2.color = sf::Color::Red;

		hills.push_back(v1);
		hills.push_back(v2);
	}

	std::vector<float> prevState(4, 0.0f);

	sf::Image image;

	image.create(16, 16);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// ---------------------------- Fitness ------------------------------

		prevFitness = fitness;
		fitness = (std::sin(position * 3.0f) + 1.0f) * 0.5f;

		// ------------------------------ AI -------------------------------

		float dFitness = (fitness - prevFitness) * 100.0f;

		std::vector<float> state(4);

		state[0] = 1.2f * (position + 0.52f);
		state[1] = velocity * 15.0f;
		state[2] = prevState[2];
		state[3] = prevState[3];

		std::vector<float> action(4);
		std::vector<bool> actionMask(4, false);

		actionMask[2] = true;
		actionMask[3] = true;

		htm.step(fitness, state, actionMask, action, 0.5f, 16, 0.1f, 0.9f, 0.0f, 0.1f, 16.0f, 1.0f, 8, 16.0f, 16.0f, 16.0f, 0.0001f, 0.01f, 0.01f, 0.01f, 0.1f, 0.1f, 0.0f, 0.05f, 0.5f, 0.99f, 0.97f, 1.0f, 0.05f, 0.05f, generator);

		float actionf;

		prevState[2] = action[2];
		prevState[3] = action[3];

		actionf = (prevState[2]) * 1.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
			actionf = -1.0f;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
			actionf = 1.0f;

		velocity += (-velocity * 0.01f + actionf * 0.001f + std::cos(3.0f * position) * -0.0025f) * dt / 0.017f;
		position += velocity * dt / 0.017f;

		// ---------------------------- Rendering ----------------------------

		window.clear(sf::Color::White);

		// Draw hills
		window.draw(&hills[0], hills.size(), sf::Lines);

		cartSprite.setPosition(sf::Vector2f(static_cast<float>(window.getSize().x) * 0.5f + pixelsPerMeter * position, static_cast<float>(window.getSize().y) * 0.5f + pixelsPerMeter * -0.333f * std::sinf(3.0f * position)));

		float slope = std::cos(3.0f * position);

		float angle = std::atan(slope);

		cartSprite.setRotation(360.0f - 180.0f / static_cast<float>(std::_Pi) * angle);

		window.draw(cartSprite);

		for (int x = 0; x < 16; x++)
		for (int y = 0; y < 16; y++) {
			sf::Color c;

			c.r = htm.getRegion().getColumn(x, y)._prediction * 255.0f;
			c.g = 0;
			c.b = 0;

			image.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setPosition(800.0f - 256.0f, 0.0f);

		s.setScale(256.0f / image.getSize().x, 256.0f / image.getSize().y);

		s.setTexture(t);

		window.draw(s);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();
	} while (!quit);
}*/

/*const size_t entrySize = 30; // 31 if you include the weight

struct DataEntry {
	std::array<double, entrySize> _data;

	unsigned int _eventID;

	double _weight;

	bool _s;
};

int main() {
	std::mt19937 generator(time(nullptr));

	std::ifstream fromFile("Resources/training/training.csv");

	if (!fromFile.is_open()) {
		std::cerr << "Could not find training data!" << std::endl;
	}

	std::array<bool, entrySize> hasNulls;

	hasNulls.assign(false);

	std::array<double, entrySize> mins;

	mins.assign(999999.0);

	std::array<double, entrySize> maxs;

	maxs.assign(-999999.0);

	std::string line;

	std::getline(fromFile, line);

	std::vector<DataEntry> entries;

	int t = 0;

	while (!fromFile.eof() && fromFile.good()) {
		std::getline(fromFile, line);

		if (line == "")
			continue;
		
		DataEntry entry;

		std::istringstream is(line);

		std::string value;

		std::getline(is, value, ',');

		entry._eventID = std::stoi(value) - 100000;

		for (size_t i = 0; i < entrySize; i++) {
			std::getline(is, value, ',');

			entry._data[i] = std::stof(value);

			if (entry._data[i] == -999.0) {
				hasNulls[i] = true;
			}
			else {
				mins[i] = std::min(mins[i], entry._data[i]);
				maxs[i] = std::max(maxs[i], entry._data[i]);
			}
		}

		std::getline(is, value, ',');

		entry._weight = std::stof(value);

		std::getline(is, value, ',');

		entry._s = value == "s";

		entries.push_back(entry);

		t++;
	}

	int numNulls = 0;

	for (size_t i = 0; i < entrySize; i++)
	if (hasNulls[i])
		numNulls++;

	// Normalize the data
	for (size_t i = 0; i < entries.size(); i++) {
		for (size_t j = 0; j < entrySize; j++) {
			if (entries[i]._data[j] != -999.0)
				entries[i]._data[j] = (entries[i]._data[j] - mins[j]) / (maxs[j] - mins[j]);
		}
	}

	deep::DBN dbn;

	//nn::FeedForwardNeuralNetwork ffnn;

	//ffnn.createRandom(entrySize + numNulls, 1, 1, 60, -0.1f, 0.1f, generator);

	std::vector<size_t> rbmHiddenSizes(2);

	rbmHiddenSizes[0] = 60;
	rbmHiddenSizes[1] = 60;

	dbn.createRandom(entrySize + numNulls, 1, rbmHiddenSizes, -0.1, 0.1, generator);

	double rbmAlpha = 0.1;
	double ffnnAlphaTune = 0.05;
	double alphaMultiplier = 1.0;
	double momentum = 0.0;
	double weightDecayMultiplier = 0.999;
	size_t miniBatchSize = 1;

	// Pre-train layers
	for (size_t l = 0; l < rbmHiddenSizes.size(); l++) {
		std::cout << "Pre-training layer " << l + 1 << std::endl;

		for (size_t i = 0; i < 1000000; i++) {
			// Select random
			std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

			int entryIndex = distEntry(generator);

			std::vector<double> data(entrySize + numNulls);

			size_t nIndex = 0;

			for (size_t j = 0; j < entrySize; j++) {
				if (hasNulls[j]) {
					if (entries[entryIndex]._data[j] == -999.0) {
						data[entrySize + nIndex] = 1.0;
						data[j] = 0.0f;
					}
					else {
						data[entrySize + nIndex] = 0.0;
						data[j] = entries[entryIndex]._data[j];
					}

					nIndex++;
				}
				else {
					data[j] = entries[entryIndex]._data[j];
				}

				assert(data[j] >= 0.0 && data[j] <= 1.0);
			}

			// Execute layers up till here
			std::vector<double> input;

			dbn.getOutputMeanThroughLayers(l, data, input);

			dbn.trainLayerUnsupervised(l, input, rbmAlpha, generator);

			if (i % 1000 == 0)
				std::cout << "Pre-training iteration " << i + 1 << std::endl;
		}
	}

	dbn.prepareForGradientDescent();

	// Fine tuning
	for (size_t i = 0; i < 1000000; i++) {
		// Select random
		std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

		int entryIndex = distEntry(generator);

		std::vector<double> data(entrySize + numNulls);

		size_t nIndex = 0;

		for (size_t j = 0; j < entrySize; j++) {
			if (hasNulls[j]) {
				if (entries[entryIndex]._data[j] == -999.0) {
					data[entrySize + nIndex] = 1.0;
					data[j] = 0.0f;
				}
				else {
					data[entrySize + nIndex] = 0.0;
					data[j] = entries[entryIndex]._data[j];
				}

				nIndex++;
			}
			else {
				data[j] = entries[entryIndex]._data[j];
			}

			assert(data[j] >= 0.0 && data[j] <= 1.0);
		}

		std::vector<double> output;

		dbn.execute(data, output);

		std::vector<double> target(1);

		target[0] = entries[entryIndex]._s ? 1.0 : -1.0;

		dbn.getError(target);

		//dbn.signError();

		dbn.accumulateGradient();

		if ((i % miniBatchSize) == (miniBatchSize - 1)) {
			dbn.moveAlongAccumulatedGradient(ffnnAlphaTune);

			dbn.decayWeights(weightDecayMultiplier);
		}

		if (i % 1000 == 0)
			std::cout << "Fine-tuning iteration " << i + 1 << std::endl;
	}

	int numAccurate = 0;

	// Train
	int numTestTrials = entries.size();

	float sRate = 0.0;
	float bRate = 0.0;

	for (size_t i = 0; i < numTestTrials; i++) {
		// Select random
		//std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

		size_t entryIndex = i;

		std::vector<double> data(entrySize + numNulls);

		size_t nIndex = 0;

		for (size_t j = 0; j < entrySize; j++) {
			if (hasNulls[j]) {
				if (entries[entryIndex]._data[j] == -999.0) {
					data[entrySize + nIndex] = 1.0;
					data[j] = 0.0f;
				}
				else {
					data[entrySize + nIndex] = 0.0;
					data[j] = entries[entryIndex]._data[j];
				}

				nIndex++;
			}
			else {
				data[j] = entries[entryIndex]._data[j];
			}

			//assert(data[j] >= 0.0 && data[j] <= 1.0);
		}

		std::vector<double> output;

		dbn.execute(data, output);

		int label = output[0] > 0.0 ? 1 : 0;

		if ((label == 1) == entries[entryIndex]._s) {
			std::cout << entryIndex << ": Accurate. Output: " << output[0] << " Result: " << (entries[entryIndex]._s ? "s" : "b") << std::endl;
		}
		else {
			std::cout << entryIndex << ": Innaccurate. Output: " << output[0] << " Result: " << (entries[entryIndex]._s ? "s" : "b") << std::endl;
		}

		if ((label == 1) == entries[entryIndex]._s) {
			numAccurate++;
		}

		if (label == 1) { // Predicted s
			if (entries[entryIndex]._s)
				sRate += entries[entryIndex]._weight;
			else
				bRate += entries[entryIndex]._weight;
		}

		if (i % 1000 == 0)
			std::cout << "Testing iteration " << i + 1 << std::endl;
	}

	int numInnaccurate = numTestTrials - numAccurate;

	std::cout << "Error: " << numInnaccurate / static_cast<float>(numTestTrials)* 100.0 << "%" << std::endl;

	std::cout << "AMS: " << std::sqrt(2.0 * ((sRate + bRate + 10.0) * std::log(1.0 + sRate / (bRate + 10.0)) - sRate)) << std::endl;

	system("pause");

	std::cout << "Loading test data..." << std::endl;

	std::ifstream fromFileTest("Resources/test/test.csv");

	if (!fromFileTest.is_open()) {
		std::cerr << "Could not find test data!" << std::endl;
	}


	std::getline(fromFileTest, line);

	entries.clear();

	t = 0;

	while (!fromFileTest.eof() && fromFileTest.good()) {
		std::getline(fromFileTest, line);

		if (line == "")
			continue;

		DataEntry entry;

		std::istringstream is(line);

		std::string value;

		std::getline(is, value, ',');

		entry._eventID = std::stoi(value);

		for (size_t i = 0; i < entrySize; i++) {
			std::getline(is, value, ',');

			entry._data[i] = std::stof(value);
		}

		entries.push_back(entry);

		t++;
	}

	fromFileTest.close();

	// Normalize the data
	for (size_t i = 0; i < entries.size(); i++) {
		for (size_t j = 0; j < entrySize; j++) {
			if (entries[i]._data[j] != -999.0)
				entries[i]._data[j] = (entries[i]._data[j] - mins[j]) / (maxs[j] - mins[j]);
		}
	}

	std::cout << "Number of entries: " << entries.size() << std::endl;;

	system("pause");

	// Output file
	std::cout << "Evaluating test data..." << std::endl;

	std::ofstream evaluationToFile("evaluation.txt");

	evaluationToFile << "EventId,RankOrder,Class" << std::endl;

	for (size_t i = 0; i < entries.size(); i++) {
		// Select random
		//std::uniform_int_distribution<int> distEntry(0, entries.size() - 1);

		size_t entryIndex = i;

		std::vector<double> data(entrySize + numNulls);

		size_t nIndex = 0;

		for (size_t j = 0; j < entrySize; j++) {
			if (hasNulls[j]) {
				if (entries[entryIndex]._data[j] == -999.0) {
					data[entrySize + nIndex] = 1.0;
					data[j] = 0.0f;
				}
				else {
					data[entrySize + nIndex] = 0.0;
					data[j] = entries[entryIndex]._data[j];
				}

				nIndex++;
			}
			else {
				data[j] = entries[entryIndex]._data[j];
			}

			//assert(data[j] >= 0.0 && data[j] <= 1.0);
		}

		std::vector<double> output;

		dbn.execute(data, output);

		int label = output[0] > 0.0 ? 1 : 0;

		//size_t rankOrder = static_cast<size_t>(std::min(1.0, std::max(0.0, output[0] * 0.5 + 0.5)) * 550000.0) + 1;

		evaluationToFile << entries[entryIndex]._eventID << "," << (i + 1) << "," << (label == 1 ? "s" : "b") << std::endl;

		if (i % 1000 == 0)
			std::cout << "Evaluated iteration " << i + 1 << std::endl;
	}

	evaluationToFile.close();

	std::cout << "Done." << std::endl;

	system("pause");

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	sf::Image img;

	img.loadFromFile("testImage.png");

	deep::ConvNet2D convNet;

	std::vector<deep::ConvNet2D::LayerPairDesc> descs(2);

	convNet.createRandom(img.getSize().x, img.getSize().y, 1, descs, -0.1f, 0.1f, generator);

	float bInv = 1.0f / 255.0f;

	for (size_t x = 0; x < img.getSize().x; x++)
	for (size_t y = 0; y < img.getSize().y; y++) {
		float greyscale = ((img.getPixel(x, y).r * bInv + img.getPixel(x, y).g * bInv + img.getPixel(x, y).b * bInv) * 0.3333f - 0.5f) * 6.0f;

		convNet.setInput(x, y, 0, greyscale);
	}

	sf::RenderWindow window;
	window.create(sf::VideoMode(convNet.getOutputWidth() * 8, convNet.getOutputHeight() * 8), "ConvNetRBM", sf::Style::Default);

	for (size_t i = 0; i < 50; i++) {
		convNet.activateAndLearn(0.01f, generator);

		sf::Image result;

		result.create(convNet.getOutputWidth(), convNet.getOutputHeight());

		for (size_t x = 0; x < convNet.getOutputWidth(); x++)
		for (size_t y = 0; y < convNet.getOutputHeight(); y++) {
			sf::Color c;

			c.r = 255.0f * std::min(1.0f, std::max(0.0f, ((convNet.getOutput(x, y, 0) - 0.5f) * 2.0f + 0.5f)));;
			c.g = 255.0f * std::min(1.0f, std::max(0.0f, ((convNet.getOutput(x, y, 1) - 0.5f) * 2.0f + 0.5f)));;
			c.b = 255.0f * std::min(1.0f, std::max(0.0f, ((convNet.getOutput(x, y, 2) - 0.5f) * 2.0f + 0.5f)));;

			result.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(result);

		sf::Sprite s;

		s.setScale(8.0f, 8.0f);

		s.setTexture(t);

		window.draw(s);

		window.display();
	}

	return 0;
}*/

/*float boostFunction(float active, float minimum) {
	return (1.0f - minimum) + std::max(0.0f, -(minimum - active));
}

int main() {
	std::mt19937 generator(time(nullptr));

	std::function<float(float, float)> boostFunc = std::bind(boostFunction, std::placeholders::_1, std::placeholders::_2);

	htm::Region region;

	region.createRandom(40, 40, 8, 6, 0, 40, 40, 5, 0.02f, 2.0f, -0.02f, 0.301f, 0.1f, generator);

	sf::RenderWindow window;

	window.create(sf::VideoMode(1200, 600), "HTM", sf::Style::Default);

	window.setFramerateLimit(40);

	float squareSize = window.getSize().y / 40.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.1f;

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::Image image;

	image.create(40, 40);

	sf::Image reconstruction;

	reconstruction.create(40, 40);

	std::vector<bool> input(40 * 40, false);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	float timer = 0.0f;
	float time = PI * 2.0f;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::F))
			window.setFramerateLimit(600);
		else
			window.setFramerateLimit(40);

		// Clear input
		for (int i = 0; i < input.size(); i++)
			input[i] = false;

		// Set one dot
		int dotX = std::round(20.0f + std::cos(timer) * 16.0f);
		int dotY = std::round(20.0f + std::sin(timer * 2.0f) * 8.0f);

		for (int dx = -2; dx <= 2; dx++)
		for (int dy = -2; dy <= 2; dy++) {
			int x = dotX + dx;
			int y = dotY + dy;

			if (x >= 0 && y >= 0 && x < 40 && y < 40)
				input[x + 40 * y] = true;
		}

		region.stepBegin();
		
		region.spatialPooling(input, 0.3f, 3.0f, 18, 0.02f, 0.015f, 0.01f, 0.05f, 0.05f, 0.015f, boostFunc);

		region.temporalPoolingLearn(0.3f, 4, 1, 10, 32, 0.02f, 0.015f, 0.301f, 6, generator);

		std::vector<bool> reconstructionData;

		int steps = 2;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3))
			steps = 3;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4))
			steps = 4;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num5))
			steps = 5;

		region.getReconstructionAtTime(reconstructionData, 3.0f, 0.3f, steps);

		
		// ---------------------------- Rendering ----------------------------

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			for (int x = 0; x < 40; x++)
			for (int y = 0; y < 40; y++) {
				sf::Color c;

				if (input[x + 40 * y])
					c = sf::Color::Blue;
				else
					c = sf::Color::White;

				image.setPixel(x, y, c);
			}
		}
		else {
			for (int x = 0; x < 40; x++)
			for (int y = 0; y < 40; y++) {
				sf::Color c;

				if (region.getColumn(x, y).isActive())
					c = sf::Color::Red;
				else
					c = sf::Color::White;

				image.setPixel(x, y, c);
			}
		}

		for (int x = 0; x < 40; x++)
		for (int y = 0; y < 40; y++) {
			sf::Color c;

			if (reconstructionData[x + y * 40])
				c = sf::Color::Red;
			else
				c = sf::Color::White;

			reconstruction.setPixel(x, y, c);
		}

		sf::Texture t;

		t.loadFromImage(image);

		sf::Sprite s;

		s.setTexture(t);

		s.setScale(squareSize, squareSize);

		window.draw(s);

		sf::Texture t2;

		t2.loadFromImage(reconstruction);

		sf::Sprite s2;

		s2.setPosition(600, 0);

		s2.setTexture(t2);

		s2.setScale(squareSize, squareSize);

		window.draw(s2);

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		timer += dt * 1.0f;

		if (timer >= time)
			timer -= time;

	} while (!quit);

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	lstm::LSTMG lstmg;

	lstmg.createRandomLayered(1, 1, 1, 2, 1, 4, -0.1f, 0.1f, generator);

	float prevY = 0.0f;

	for (int i = 0; i < 2000; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.2f) {
		float y = std::sin(x);

		lstmg.setInput(0, 0.0f);

		lstmg.step(true);

		lstmg.getDeltas(std::vector<float>(1, y), 0.0f, true);

		lstmg.moveAlongDeltas(0.01f);

		prevY = y;
	}

	prevY = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.2f) {
		float y = std::sin(x);

		lstmg.setInput(0, 0.0f);

		lstmg.step(true);

		std::cout << lstmg.getOutput(0) << std::endl;

		prevY = y;
	}

	return 0;
}*/

/*float testFunc(float x) {
	return std::sin(x * 3.0f);
}

void addPlotData(std::ostream &os, int plotIndex, const std::vector<std::tuple<float, float>> &data) {
	os << "local c" << plotIndex << " = { " << std::endl;

	for (int i = 0; i < data.size(); i++) {
		os << "    { " << std::get<0>(data[i]) << ", " << std::get<1>(data[i]) << " }";

		if (i != data.size() - 1)
			os << ",";

		os << std::endl;
	}

	os << "}" << std::endl << std::endl;
}

int main() {
	int iter = 64;

	std::mt19937 generator(time(nullptr));

	std::ofstream toFile("output.txt");

	std::vector<std::tuple<float, float>> data;

	data.clear();

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		data.push_back(std::make_tuple(x, y));
	}

	addPlotData(toFile, 1, data);

	data.clear();

	//std::cout << "SDR RBF:" << std::endl;

	chtm::CHTMRegion htm;

	htm.createRandom(2, 2, 16, 16, 3, 2, 3, 1, -0.1f, 0.1f, 0.01f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, -0.1f, 0.1f, generator);

	for (int i = 0; i < iter; i++) {
		float prevY = 0.0f;

		for (float x = 0.0f; x < 6.28f; x += 0.1f) {
			float y = testFunc(x);

			std::vector<float> input(4, x);
			std::vector<float> output(1);
			std::vector<float> target(1, y);

			htm.stepBegin();

			htm.getOutput(input, output, 3, 32.0f, 8.0f, 8.0f, generator);

			htm.learn(input, output, target, 0.01f, 0.05f, 0.05f, 2.0f, 0.01f, 0.2f, 0.01f);

			prevY = y;
		}

		std::cout << i << std::endl;
	}

	float error = 0.0f;

	sf::Image img;

	img.create(16, 16);

	float prevY = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(4, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		htm.stepBegin();

		htm.getOutput(input, output, 3, 32.0f, 8.0f, 8.0f, generator);

		if (x == 0.0f) {
			for (int x = 0; x < 16; x++)
			for (int y = 0; y < 16; y++) {
				sf::Color c = sf::Color::Black;
				c.r = 255.0f * htm.getColumn(x, y)._output;

				img.setPixel(x, y, c);
			}
		}

		error += std::abs(output[0] - y);

		//sdrrbf.update(input, output, target, 0.05f, 0.05f, 0.05f, 1.0f);

		std::cout << x << " " << output[0] << std::endl;

		data.push_back(std::make_tuple(x, output[0]));

		prevY = y;

		system("pause");
	}

	addPlotData(toFile, 2, data);

	data.clear();

	img.saveToFile("region.png");

	//std::cout << "Error: " << error << std::endl;

	//system("pause");

	//std::cout << "MLP:" << std::endl;
	deep::FA fans;

	fans.createRandom(1, 1, 1, 256, 0.1f, generator);

	std::uniform_real_distribution<float> angleDist(0.0f, 6.28f);

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {

		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fans.clearGradient();

		fans.process(input, output);

		fans.accumulateGradient(input, target);

		fans.moveAlongGradientRMS(0.05f, 0.01f, 0.1f, 0.0f, 0.0f);
	}

	error = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fans.process(input, output);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 3, data);

	data.clear();

	deep::SharpFA fa;

	fa.createRandom(1, 1, 1, 256, 0.1f, generator);

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {

		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fa.clearGradient();

		fa.process(input, output, 0.5f, 3);

		fa.accumulateGradient(input, target);

		fa.moveAlongGradientRMS(0.05f, 0.01f, 0.1f, 0.0f, 0.0f);
	}

	error = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fa.process(input, output, 0.5f, 3);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 4, data);

	data.clear();

	deep::DBN dbn;

	dbn.createRandom(1, 1, std::vector<int>(1, 256), -0.1f, 0.1f, generator);

	for (int i = 0; i < 10000; i++) {
		float x2 = angleDist(generator);

		float y = testFunc(x2);

		std::vector<float> input(1, x2);
		std::vector<float> output(1);
		std::vector<float> mean(16);
		std::vector<float> target(1, y);

		dbn.trainLayerUnsupervised(0, input, 0.01f, generator);
	}

	error = 0.0f;

	dbn.prepareForGradientDescent();

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		dbn.execute(input, output);
	
		dbn.getError(target);

		dbn.accumulateGradient();

		dbn.moveAlongAccumulatedGradient(0.001f);
	}
	
	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		dbn.execute(input, output);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 5, data);

	data.clear();

	deep::FA fanss;

	fanss.createRandom(1, 1, 1, 256, 0.1f, generator);

	for (int i = 0; i < iter; i++)
	for (float x = 0.0f; x < 6.28f; x += 0.01f) {

		float x2 = angleDist(generator);

		float y = testFunc(x2);

		std::vector<float> input(1, x2);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fanss.clearGradient();

		fanss.process(input, output);

		fanss.accumulateGradient(input, target);

		fanss.moveAlongGradientRMS(0.05f, 0.01f, 0.1f, 0.0f, 0.0f);
	}

	error = 0.0f;

	for (float x = 0.0f; x < 6.28f; x += 0.1f) {
		float y = testFunc(x);

		std::vector<float> input(1, x);
		std::vector<float> output(1);
		std::vector<float> target(1, y);

		fanss.process(input, output);

		error += std::abs(output[0] - y);

		data.push_back(std::make_tuple(x, output[0]));

		std::cout << output[0] << std::endl;
	}

	addPlotData(toFile, 6, data);

	data.clear();

	toFile.close();

	//std::cout << "Error: " << error << std::endl;

	//system("pause");

	//c1.prepare(sf::Vector2f(0.0f, 6.28f), sf::Vector2f(-2.0f, 2.0f));

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	sf::RenderWindow renderWindow;

	renderWindow.create(sf::VideoMode(512, 512), "Circles", sf::Style::Default);

	sf::Clock clock;

	bool quit = false;

	bool aPressedLastFrame = false;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (renderWindow.pollEvent(windowEvent)) {
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A) && !aPressedLastFrame) {

		}

		aPressedLastFrame = sf::Keyboard::isKeyPressed(sf::Keyboard::A);

	} while (!quit);

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	float reward = 0.0f;
	float prevReward = 0.0f;

	float initReward = 0.0f;

	float totalReward = 0.0f;

	sf::RenderWindow window;

	window.create(sf::VideoMode(1600, 600), "Pole Balancing");

	window.setVerticalSyncEnabled(true);

	//window.setFramerateLimit(60);

	// -------------------------- Load Resources --------------------------

	sf::Texture backgroundTexture;
	sf::Texture cartTexture;
	sf::Texture poleTexture;

	backgroundTexture.loadFromFile("Resources/background.png");
	cartTexture.loadFromFile("Resources/cart.png");
	poleTexture.loadFromFile("Resources/pole.png");

	sf::Texture inputCartTexture;
	sf::Texture inputPoleTexture;

	inputCartTexture.loadFromFile("Resources/inputCart.png");
	inputPoleTexture.loadFromFile("Resources/inputPole.png");

	// --------------------------------------------------------------------

	sf::Sprite backgroundSprite;
	sf::Sprite cartSprite;
	sf::Sprite poleSprite;

	backgroundSprite.setTexture(backgroundTexture);
	cartSprite.setTexture(cartTexture);
	poleSprite.setTexture(poleTexture);

	backgroundSprite.setPosition(sf::Vector2f(0.0f, 0.0f));

	cartSprite.setOrigin(sf::Vector2f(static_cast<float>(cartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(cartSprite.getTexture()->getSize().y)));
	poleSprite.setOrigin(sf::Vector2f(static_cast<float>(poleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(poleSprite.getTexture()->getSize().y)));

	sf::Sprite inputCartSprite;
	sf::Sprite inputPoleSprite;

	inputCartSprite.setTexture(inputCartTexture);
	inputPoleSprite.setTexture(inputPoleTexture);

	inputCartSprite.setOrigin(sf::Vector2f(static_cast<float>(inputCartSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputCartSprite.getTexture()->getSize().y)));
	inputPoleSprite.setOrigin(sf::Vector2f(static_cast<float>(inputPoleSprite.getTexture()->getSize().x) * 0.5f, static_cast<float>(inputPoleSprite.getTexture()->getSize().y)));


	// ----------------------------- Physics ------------------------------

	float pixelsPerMeter = 128.0f;
	float inputPixelsPerMeter = 8.0f;
	float poleLength = 1.0f;
	float g = -2.8f;
	float massMass = 40.0f;
	float cartMass = 2.0f;
	sf::Vector2f massPos(0.0f, poleLength);
	sf::Vector2f massVel(0.0f, 0.0f);
	float poleAngle = static_cast<float>(PI)* 0.0f;
	float poleAngleVel = 0.0f;
	float poleAngleAccel = 0.0f;
	float cartX = 0.0f;
	float cartVelX = 0.0f;
	float cartAccelX = 0.0f;
	float poleRotationalFriction = 0.008f;
	float cartMoveRadius = 1.8f;
	float cartFriction = 0.02f;
	float maxSpeed = 3.0f;

	// ---------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	float fitness = 0.0f;
	float prevFitness = 0.0f;

	float lowPassFitness = 0.0f;

	bool reverseDirection = false;

	bool trainMode = true;

	bool tDownLastFrame = false;

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	sf::Font font;

	font.loadFromFile("Resources/pixelated.ttf");

	sf::RenderTexture inputRT;

	inputRT.create(64, 32);

	sf::RenderTexture rt;

	rt.create(800, 600);

	float avgReward = 0.0f;
	float avgRewardDecay = 0.003f;

	float totalTime = 0.0f;

	float plotUpdateTimer = 0.0f;

	deep::RSARL agent;

	agent.createRandom(4, 1, 64, 5.01f / 64.0f, -0.1f, 0.1f, 0.1f, generator);

	std::vector<float> prevInput(6, 0.0f);

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		// Update fitness
		if (poleAngle < static_cast<float>(PI))
			fitness = -(static_cast<float>(PI)* 0.5f - poleAngle);
		else
			fitness = -(static_cast<float>(PI)* 0.5f - (static_cast<float>(PI)* 2.0f - poleAngle));

		fitness += static_cast<float>(PI)* 0.5f;

		//fitness = fitness - std::abs(poleAngleVel * 1.0f);

		//fitness = -std::abs(cartX);

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
			fitness = -cartX;
		else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
			fitness = cartX;

		// ------------------------------ AI -------------------------------

		float dFitness = fitness - prevFitness;

		//reward = dFitness * 5.0f;

		reward = fitness * 0.5f;

		if (totalTime == 0.0f)
			avgReward = reward;
		else
			avgReward = (1.0f - avgRewardDecay) * avgReward + avgRewardDecay * reward;

		sf::Image img = inputRT.getTexture().copyToImage();

		agent.setInput(0, cartX * 0.25f);
		agent.setInput(1, cartVelX * 0.1f);
		agent.setInput(2, std::fmod(poleAngle + static_cast<float>(PI), 2.0f * static_cast<float>(PI)) * 0.1f);
		agent.setInput(3, poleAngleVel * 0.1f);

		agent.step(reward, 10, 20, 0.0f, 0.01f, 0.005f, 0.0f, 0.3f, 0.01f, 0.5f, 1.0f, 1.0f, 1.0f, 0.1f, 0.7f, 0.992f, 0.1f, generator);

		float dir = agent.getOutput(0);

		float agentForce = 4000.0f * dir;

		prevFitness = fitness;

		// ---------------------------- Physics ----------------------------

		float pendulumCartAccelX = cartAccelX;

		if (cartX < -cartMoveRadius)
			pendulumCartAccelX = 0.0f;
		else if (cartX > cartMoveRadius)
			pendulumCartAccelX = 0.0f;

		poleAngleAccel = pendulumCartAccelX * std::cos(poleAngle) + g * std::sin(poleAngle);
		poleAngleVel += -poleRotationalFriction * poleAngleVel + poleAngleAccel * dt;
		poleAngle += poleAngleVel * dt;

		massPos = sf::Vector2f(cartX + std::cos(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength, std::sin(poleAngle + static_cast<float>(PI)* 0.5f) * poleLength);

		float force = 0.0f;

		if (std::abs(cartVelX) < maxSpeed) {
			force = std::max(-4000.0f, std::min(4000.0f, agentForce));

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left))
				force = -4000.0f;

			if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
				force = 4000.0f;
		}

		if (cartX < -cartMoveRadius) {
			cartX = -cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}
		else if (cartX > cartMoveRadius) {
			cartX = cartMoveRadius;

			cartAccelX = -cartVelX / dt;
			cartVelX = -0.5f * cartVelX;
		}

		cartAccelX = 0.25f * (force + massMass * poleLength * poleAngleAccel * std::cos(poleAngle) - massMass * poleLength * poleAngleVel * poleAngleVel * std::sin(poleAngle)) / (massMass + cartMass);
		cartVelX += -cartFriction * cartVelX + cartAccelX * dt;
		cartX += cartVelX * dt;

		poleAngle = std::fmod(poleAngle, (2.0f * static_cast<float>(PI)));

		if (poleAngle < 0.0f)
			poleAngle += static_cast<float>(PI)* 2.0f;

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
			if (!tDownLastFrame) {
				trainMode = !trainMode;
			}

			tDownLastFrame = true;
		}
		else
			tDownLastFrame = false;

		// ---------------------------- Rendering ----------------------------

		// Render to input buffer
		inputRT.clear();

		inputCartSprite.setPosition(sf::Vector2f(inputRT.getSize().x * 0.5f + inputPixelsPerMeter * cartX, inputRT.getSize().y * 0.5f + 4.0f));

		inputRT.draw(inputCartSprite);

		inputPoleSprite.setPosition(inputCartSprite.getPosition() + sf::Vector2f(0.0f, -4.0f));
		inputPoleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI)+180.0f);

		inputRT.draw(inputPoleSprite);

		inputRT.display();

		window.clear();

		window.draw(backgroundSprite);

		cartSprite.setPosition(sf::Vector2f(800.0f * 0.5f + pixelsPerMeter * cartX, 600.0f * 0.5f + 3.0f));

		window.draw(cartSprite);

		poleSprite.setPosition(cartSprite.getPosition() + sf::Vector2f(0.0f, -45.0f));
		poleSprite.setRotation(poleAngle * 180.0f / static_cast<float>(PI)+180.0f);

		window.draw(poleSprite);

		// Draw hidden states
		sf::Vector2f hiddenCenter(1200.0f, 300.0f);

		float sScalar = 1.5f;

		int rowSize = 8;

		int rCount = 0;
		int row = -4;

		while (rCount < agent.getRSA().getNumHiddenNodes()) {
			for (int i = 0; i < rowSize; i++) {
				sf::CircleShape circle;

				circle.setPosition(hiddenCenter + sf::Vector2f((i - rowSize / 2) * 18.0f * sScalar, row * 18.0f * sScalar));

				circle.setRadius(8.0f * sScalar);

				circle.setFillColor(sf::Color::White);

				circle.setOrigin(8.0f * sScalar, 8.0f * sScalar);

				window.draw(circle);

				circle.setRadius(7.0f * sScalar);

				circle.setOrigin(7.0f * sScalar, 7.0f * sScalar);

				int s = (agent.getRSA().getHiddenNodeState(rCount)) * 255;

				circle.setFillColor(sf::Color(s, s, s, 255));

				window.draw(circle);

				rCount++;
			}

			row++;
		}

		// -------------------------------------------------------------------

		window.display();

		//dt = clock.getElapsedTime().asSeconds();

		totalTime += dt;
		plotUpdateTimer += dt;
	} while (!quit);

	return 0;
}*/

	/*deep::FERL ferl;

	ferl.createRandom(2, 2, 12, 0.1f, generator);

	std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

	int tunnelLength = 4;

	float reward = 0.0f;

	for (int e = 0; e < 10000; e++) {
		int x = 1;

		bool sideIsUp = dist01(generator) < 0.5f;

		for (int t = 0; t < 8; t++) {
			std::vector<float> input(2);

			if (t == 0)
				input[0] = sideIsUp ? 1.0f : -1.0f;
			else
				input[0] = 0.0f;

			if (x == tunnelLength)
				input[1] = 1.0f;
			else
				input[1] = 0.0f;

			std::vector<float> action(2);

			ferl.step(input, action, reward, 0.5f, 0.9f, 0.9f, 1.0f, 32, 4, 0.05f, 0.15f, 0.05f, 800, 400, 0.05f, 0.0f, generator);

			reward *= 0.5f;

			if (x == tunnelLength) {
				if (action[0] < -0.333f)
					x--;

				if (action[1] > 0.333f) {
					if (sideIsUp)
						reward = 1.0f;
					else
						reward = -0.5f;

					break;
				}
				else if (action[1] < -0.333f) {
					if (!sideIsUp)
						reward = 1.0f;
					else
						reward = -0.5f;

					break;
				}
			}
			else if (x == 1) {
				if (action[0] > 0.333f)
					x++;
			}
			else {
				if (action[0] > 0.333f)
					x++;
				else if (action[0] < -0.333f)
					x--;
			}
		}
	}

	// Tests
	int successes = 0;
	int failures = 0;

	for (int e = 0; e < 1000; e++) {
		int x = 1;

		bool sideIsUp = dist01(generator) < 0.5f;

		for (int t = 0; t < 8; t++) {
			std::vector<float> input(2);

			if (t == 0)
				input[0] = sideIsUp ? 1.0f : -1.0f;
			else
				input[0] = 0.0f;

			if (x == tunnelLength)
				input[1] = 1.0f;
			else
				input[1] = 0.0f;

			std::vector<float> action(2);

			ferl.step(input, action, reward, 0.5f, 0.9f, 0.9f, 1.0f, 32, 4, 0.05f, 0.15f, 0.05f, 800, 400, 0.05f, 0.0f, generator);

			reward *= 0.5f;

			if (x == tunnelLength) {
				if (action[0] < -0.333f)
					x--;

				if (action[1] > 0.333f) {
					if (sideIsUp) {
						reward = 1.0f;
						successes++;
					}
					else {
						reward = -0.5f;
						failures++;
					}

					break;
				}
				else if (action[1] < -0.333f) {
					if (!sideIsUp) {
						reward = 1.0f;
						successes++;
					}
					else {
						reward = -0.5f;
						failures++;
					}

					break;
				}
			}
			else if (x == 1) {
				if (action[0] > 0.333f)
					x++;
			}
			else {
				if (action[0] > 0.333f)
					x++;
				else if (action[0] < -0.333f)
					x--;
			}
		}
	}

	std::cout << successes << " " << failures << std::endl;
}*/

/*struct Image {
	std::vector<unsigned char> _image;
};

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void getMNISTImages(std::vector<Image> &images, const std::string &filename, int numUse) {
	std::ifstream file(filename, std::ios::binary);

	if (file.is_open()) {
		int magicNumber = 0;
		int nImages = 0;
		int nRows = 0;
		int nCols = 0;

		file.read((char*)&magicNumber, sizeof(int));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&nImages, sizeof(int));
		nImages = reverseInt(nImages);
		file.read((char*)&nRows, sizeof(int));
		nRows = reverseInt(nRows);
		file.read((char*)&nCols, sizeof(int));
		nCols = reverseInt(nCols);

		assert(numUse <= nImages);

		images.resize(numUse);

		for (int i = 0; i < numUse; i++) {
			assert(file.good());

			images[i]._image.resize(nRows * nCols);

			file.read((char*)&(images[i]._image[0]), sizeof(unsigned char) * images[i]._image.size());
		}
	}

	file.close();
}

void getMNISTLabels(std::vector<unsigned char> &labels, const std::string &filename, int numUse) {
	std::ifstream file(filename, std::ios::binary);

	if (file.is_open()) {
		int magicNumber = 0;
		int nLabels = 0;
		int nRows = 0;
		int nCols = 0;

		file.read((char*)&magicNumber, sizeof(magicNumber));
		magicNumber = reverseInt(magicNumber);
		file.read((char*)&nLabels, sizeof(nLabels));
		nLabels = reverseInt(nLabels);

		assert(numUse <= nLabels);

		labels.resize(numUse);

		for (int i = 0; i < numUse; i++) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(unsigned char));

			labels[i] = temp;
		}
	}

	file.close();
}

int main() {
	std::vector<Image> trainingImages;
	std::vector<unsigned char> trainingLabels;

	getMNISTImages(trainingImages, "MNIST/train-images.idx3-ubyte", 60000);
	getMNISTLabels(trainingLabels, "MNIST/train-labels.idx1-ubyte", 60000);

	std::vector<Image> testImages;
	std::vector<unsigned char> testLabels;

	getMNISTImages(testImages, "MNIST/t10k-images.idx3-ubyte", 10000);
	getMNISTLabels(testLabels, "MNIST/t10k-labels.idx1-ubyte", 10000);

	// Get list of odd and even examples
	std::vector<int> oddIndices;
	std::vector<int> evenIndices;

	for (int i = 0; i < trainingLabels.size(); i++) {
		if (trainingLabels[i] % 2 == 0)
			evenIndices.push_back(i);
		else
			oddIndices.push_back(i);
	}

	rbf::SDRRBFNetwork sdrrbfnet;

	std::mt19937 generator(time(nullptr));

	std::vector<rbf::SDRRBFNetwork::LayerDesc> layerDescs(3);

	layerDescs[0]._rbfWidth = 56;
	layerDescs[0]._rbfHeight = 56;
	layerDescs[0]._receptiveRadius = 4;
	layerDescs[0]._inhibitionRadius = 3;
	layerDescs[0]._localActivity = 4.0f;
	layerDescs[0]._outputIntensity = 0.05f;
	layerDescs[0]._outputMultiplier = 0.0f;

	layerDescs[1]._rbfWidth = 40;
	layerDescs[1]._rbfHeight = 40;
	layerDescs[1]._receptiveRadius = 4;
	layerDescs[1]._inhibitionRadius = 3;
	layerDescs[1]._localActivity = 4.0f;
	layerDescs[1]._outputIntensity = 0.05f;
	layerDescs[1]._outputMultiplier = 0.5f;

	layerDescs[2]._rbfWidth = 24;
	layerDescs[2]._rbfHeight = 24;
	layerDescs[2]._receptiveRadius = 4;
	layerDescs[2]._inhibitionRadius = 3;
	layerDescs[2]._localActivity = 4.0f;
	layerDescs[2]._outputIntensity = 0.05f;
	layerDescs[2]._outputMultiplier = 0.5f;

	sdrrbfnet.createRandom(28, 28, layerDescs, 10, 0.05f, 0.95f, 0.1f, 0.3f, -0.5f, 0.5f, generator);

	std::vector<float> inputf(28 * 28);
	std::vector<float> outputf(10);

	std::uniform_int_distribution<int> selectionDist(0, trainingImages.size() - 1);
	std::uniform_int_distribution<int> selectionDistEven(0, evenIndices.size() - 1);
	std::uniform_int_distribution<int> selectionDistOdd(0, oddIndices.size() - 1);

	int totalIterUnsupervised = 1000;
	int totalIterSupervised = 1000;

	for (int i = 0; i < totalIterUnsupervised; i++) {
		int trainIndex = selectionDist(generator);

		for (int j = 0; j < trainingImages[trainIndex]._image.size(); j++)
			inputf[j] = trainingImages[trainIndex]._image[j] / 255.0f;

		sdrrbfnet.getOutput(inputf, outputf, 1.0f, 0.01f, 0.0f, 0.0f, 0.001f, 0.5f, generator);

		std::vector<float> target(10, 0.0f);

		target[trainingLabels[trainIndex]] = 1.0f;

		sdrrbfnet.updateUnsupervised(inputf, 0.05f, 0.9f, 0.2f, 1.0f, 0.0001f, 0.01f);

		if (i % 100 == 0)
			std::cout << "Iter Unsupervised: " << i << " / " << totalIterUnsupervised << std::endl;
	}

	for (int i = 0; i < totalIterSupervised; i++) {
		int trainIndex = selectionDist(generator);

		for (int j = 0; j < trainingImages[trainIndex]._image.size(); j++)
			inputf[j] = trainingImages[trainIndex]._image[j] / 255.0f;

		sdrrbfnet.getOutput(inputf, outputf, 1.0f, 0.01f, 0.0f, 0.0f, 0.001f, 0.5f, generator);

		std::vector<float> target(10, 0.0f);

		target[trainingLabels[trainIndex]] = 1.0f;

		sdrrbfnet.updateSupervised(inputf, outputf, target, 0.05f, 0.1f, 0.1f, 1.0f, 0.0001f, 0.01f);

		if (i % 100 == 0)
			std::cout << "Supervised Iter: " << i << " / " << totalIterSupervised << std::endl;
	}

	int wrongs = 0;
	int oddWrongs = 0;
	int totalOdds = 0;

	std::uniform_int_distribution<int> testDist(0, testImages.size() - 1);

	int totalIterTest = 500;

	for (int i = 0; i < totalIterTest; i++) {
		int trainIndex = i;

		for (int j = 0; j < testImages[trainIndex]._image.size(); j++)
			inputf[j] = testImages[trainIndex]._image[j] / 255.0f;

		sdrrbfnet.getOutput(inputf, outputf, 1.0f, 0.01f, 0.0f, 0.0f, 0.001f, 0.01f, generator);

		int maxIndex = 0;

		for (int j = 0; j < outputf.size(); j++)
		if (outputf[j] > outputf[maxIndex])
			maxIndex = j;

		if (maxIndex != testLabels[trainIndex])
			wrongs++;

		if (testLabels[trainIndex] % 2 == 1) {
			totalOdds++;

			if (maxIndex != testLabels[trainIndex])
				oddWrongs++;
		}

		std::cout << "Result: " << maxIndex << " Actual: " << static_cast<int>(testLabels[trainIndex]) << std::endl;
	}

	std::cout << "Total Error: " << (static_cast<float>(wrongs) / static_cast<float>(totalIterTest)) * 100.0f << std::endl;
	std::cout << "Odd Error: " << (static_cast<float>(oddWrongs) / static_cast<float>(totalOdds)) * 100.0f << std::endl;

	return 0;
}*/

/*struct Sample {
	float Dens_Lab;
	float FREQ;
	float BW;
	float AMP;
};

void loadSamples(const std::string &fileName, std::vector<Sample> &samples) {
	std::ifstream fromFile(fileName);

	samples.clear();

	std::string temp;

	// Read away header
	fromFile >> temp >> temp >> temp >> temp;

	while (fromFile.good() && !fromFile.eof()) {
		Sample s;
		
		fromFile >> s.Dens_Lab >> s.FREQ >> s.BW >> s.AMP;

		samples.push_back(s);
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	std::vector<Sample> samples;
	std::vector<float> valuePossibilities;

	const float valueTolerance = 0.001f;
	const float sparsity = 0.1f;

	loadSamples("Resources/hilmar_train/train.txt", samples);

	Sample minSamples = samples[0];
	Sample maxSamples = samples[0];

	for (int si = 1; si < samples.size(); si++) {
		minSamples.Dens_Lab = std::min(minSamples.Dens_Lab, samples[si].Dens_Lab);
		minSamples.FREQ = std::min(minSamples.FREQ, samples[si].FREQ);
		minSamples.BW = std::min(minSamples.BW, samples[si].BW);
		minSamples.AMP = std::min(minSamples.AMP, samples[si].AMP);

		maxSamples.Dens_Lab = std::max(maxSamples.Dens_Lab, samples[si].Dens_Lab);
		maxSamples.FREQ = std::max(maxSamples.FREQ, samples[si].FREQ);
		maxSamples.BW = std::max(maxSamples.BW, samples[si].BW);
		maxSamples.AMP = std::max(maxSamples.AMP, samples[si].AMP);

		// Search for similar values
		bool found = false;

		for (int i = 0; i < valuePossibilities.size(); i++) {
			if (std::abs(valuePossibilities[i] - samples[si].Dens_Lab) < valueTolerance) {
				found = true;
				break;
			}
		}

		if (!found)
			valuePossibilities.push_back(samples[si].Dens_Lab);
	}

	std::vector<int> randomizedSampleIndicies(samples.size());

	for (int i = 0; i < randomizedSampleIndicies.size(); i++)
		randomizedSampleIndicies[i] = i;

	std::shuffle(randomizedSampleIndicies.begin(), randomizedSampleIndicies.end(), generator);

	int testSize = 10;

	int testStart = randomizedSampleIndicies.size() - testSize;

	deep::SRBMFA fa;

	std::vector<deep::SRBMFA::LayerDesc> layerDescs(2);

	layerDescs[0]._numHiddenUnits = 200;
	
	layerDescs[1]._numHiddenUnits = 80;

	fa.createRandom(3, layerDescs, 1, 0.05f, generator);

	std::uniform_int_distribution<int> sampleDist(0, testStart - 1);

	int unsupervisedIterations = 80000;

	for (int t = 0; t < unsupervisedIterations; t++) {
		int si = sampleDist(generator);

		Sample &s = samples[randomizedSampleIndicies[si]];

		fa.setInput(0, (s.FREQ - minSamples.FREQ) / (maxSamples.FREQ - minSamples.FREQ));
		fa.setInput(1, (s.BW - minSamples.BW) / (maxSamples.BW - minSamples.BW));
		fa.setInput(2, (s.AMP - minSamples.AMP) / (maxSamples.AMP - minSamples.AMP));

		for (int l = 0; l < layerDescs.size(); l++) {
			fa.activate(sparsity, l);

			fa.learnRBM(l, 0.01f, 0.01f, 0.3f, 0.04f, sparsity, 1, generator);
		}

		if (t % 1000 == 0)
			std::cout << (static_cast<float>(t) / unsupervisedIterations * 100.0f) << "%" << std::endl;;
	}

	int supervisedIterations = 80000;

	for (int t = 0; t < supervisedIterations; t++) {
		int si = sampleDist(generator);

		Sample &s = samples[randomizedSampleIndicies[si]];

		fa.setInput(0, (s.FREQ - minSamples.FREQ) / (maxSamples.FREQ - minSamples.FREQ));
		fa.setInput(1, (s.BW - minSamples.BW) / (maxSamples.BW - minSamples.BW));
		fa.setInput(2, (s.AMP - minSamples.AMP) / (maxSamples.AMP - minSamples.AMP));

		fa.activate(sparsity);

		fa.setOutputUnitTarget(0, s.Dens_Lab);

		fa.learnFFNN(0.01f, 0.3f);

		//std::cout << fa << std::endl;

		if (t % 1000 == 0)
			std::cout << (static_cast<float>(t) / supervisedIterations * 100.0f) << "%" << std::endl;
	}

	float errorSum = 0.0f;

	for (int i = 0; i < testSize; i++) {
		int si = testStart + i;

		Sample &s = samples[randomizedSampleIndicies[si]];

		fa.setInput(0, (s.FREQ - minSamples.FREQ) / (maxSamples.FREQ - minSamples.FREQ));
		fa.setInput(1, (s.BW - minSamples.BW) / (maxSamples.BW - minSamples.BW));
		fa.setInput(2, (s.AMP - minSamples.AMP) / (maxSamples.AMP - minSamples.AMP));

		fa.activate(sparsity);

		int closestIndex = 0;
		float minDistance = 999999.0f;

		float find = fa.getOutputUnitOutput(0);

		for (int i = 0; i < valuePossibilities.size(); i++) {
			if (std::abs(valuePossibilities[i] - find) < minDistance) {
				minDistance = std::abs(valuePossibilities[i] - find);
				closestIndex = i;
			}
		}

		float out = valuePossibilities[closestIndex];// minDistance < 0.1f ? valuePossibilities[closestIndex] : find;

		float error = std::abs((s.Dens_Lab - out) / s.Dens_Lab);

		errorSum += error;

		std::cout << "Output: " << out << " Actual: " << s.Dens_Lab << std::endl;
	}

	errorSum /= testSize;

	std::cout << "Average Error: " << errorSum * 100.0f << std::endl;

	system("pause");

	return 0;
}*/

/*struct Sample {
	float _class;
	std::vector<float> _parameters;
};

void loadSamples(const std::string &fileName, std::vector<Sample> &samples, int parameters) {
	std::ifstream fromFile(fileName);

	samples.clear();

	std::string temp;

	// Read away header
	std::string line;
	std::getline(fromFile, line);

	while (fromFile.good() && !fromFile.eof()) {
		Sample s;

		s._parameters.resize(parameters);

		fromFile >> s._class;

		for (int i = 0; i < parameters; i++)
			fromFile >> s._parameters[i];

		samples.push_back(s);
	}
}

int main() {
	std::mt19937 generator(time(nullptr));

	std::vector<Sample> samples;
	std::vector<float> valuePossibilities;

	const float valueTolerance = 0.001f;
	const int parameters = 4;

	loadSamples("Resources/hilmar_train/train.txt", samples, parameters);

	Sample minSamples = samples[0];
	Sample maxSamples = samples[0];

	for (int si = 1; si < samples.size(); si++) {
		minSamples._class = std::min(minSamples._class, samples[si]._class);

		for (int i = 0; i < parameters; i++)
			minSamples._parameters[i] = std::min(minSamples._parameters[i], samples[si]._parameters[i]);

		for (int i = 0; i < parameters; i++)
			maxSamples._parameters[i] = std::max(maxSamples._parameters[i], samples[si]._parameters[i]);

		// Search for similar values
		bool found = false;

		for (int i = 0; i < valuePossibilities.size(); i++) {
			if (std::abs(valuePossibilities[i] - samples[si]._class) < valueTolerance) {
				found = true;
				break;
			}
		}

		if (!found)
			valuePossibilities.push_back(samples[si]._class);
	}

	std::vector<int> randomizedSampleIndicies(samples.size());

	for (int i = 0; i < randomizedSampleIndicies.size(); i++)
		randomizedSampleIndicies[i] = i;

	std::shuffle(randomizedSampleIndicies.begin(), randomizedSampleIndicies.end(), generator);

	int testSize = 300;

	int testStart = randomizedSampleIndicies.size() - testSize;

	float errorSum = 0.0f;

	for (int i = 0; i < testSize; i++) {
		int si = testStart + i;

		Sample &s = samples[randomizedSampleIndicies[si]];

		std::vector<float> weights(testStart);

		for (int j = 0; j < weights.size(); j++) {
			float dist2 = 0.0f;
			
			for (int k = 0; k < parameters; k++)
				dist2 += std::pow(s._parameters[k] - samples[randomizedSampleIndicies[j]]._parameters[k], 2);
				
			weights[j] = dist2;
		}

		int minIndex = 0;

		for (int j = 1; j < weights.size(); j++)
			if (weights[j] < weights[minIndex])
				minIndex = j;

		float out = samples[randomizedSampleIndicies[minIndex]]._class;// minDistance < 0.1f ? valuePossibilities[closestIndex] : find;

		float error = std::abs((s._class - out) / s._class);

		errorSum += error;

		std::cout << "Output: " << out << " Actual: " << s._class << std::endl;
	}

	errorSum /= testSize;

	std::cout << "Average Error: " << errorSum * 100.0f << std::endl;

	system("pause");

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	deep::RecurrentSparseAutoencoder rsa;

	const float sequence[96][4] = {
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 1.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 1.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 1.0f, 1.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 0.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 0.0f, 0.0f, 1.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 0.0f, 1.0f, 1.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 1.0f, 1.0f, 1.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 1.0f, 1.0f },
		{ 1.0f, 0.0f, 0.0f, 1.0f },
		{ 1.0f, 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f, 0.0f }
	};

	float sparsity = 5.01f / 64.0f;
	float dutyCycleDecay = 0.01f;

	rsa.createRandom(4, 64, sparsity, -0.1f, 0.1f, 0.5f, generator);

	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 16; j++) {
			rsa.stepBegin();

			rsa.setVisibleNodeState(0, sequence[j][0]);
			rsa.setVisibleNodeState(1, sequence[j][1]);
			rsa.setVisibleNodeState(2, sequence[j][2]);
			rsa.setVisibleNodeState(3, sequence[j][3]);

			rsa.learn(sparsity, 0.0f, 0.1f, 0.1f, 0.0f, 1.0f, 0.8f, 1.0f, 1.0f);

			rsa.activate(sparsity, dutyCycleDecay);
		}
	}

	float correctCount = 0.0f;
	float totalCount = 0.0f;

	rsa.clearMemory();

	for (int i = 0; i < 1; i++) {
		for (int j = 0; j < 16; j++) {
			rsa.stepBegin();

			rsa.setVisibleNodeState(0, sequence[j][0]);
			rsa.setVisibleNodeState(1, sequence[j][1]);
			rsa.setVisibleNodeState(2, sequence[j][2]);
			rsa.setVisibleNodeState(3, sequence[j][3]);

			float b0 = rsa.getVisibleNodeReconstruction(0) > 0.5f ? 1.0f : 0.0f;
			float b1 = rsa.getVisibleNodeReconstruction(1) > 0.5f ? 1.0f : 0.0f;
			float b2 = rsa.getVisibleNodeReconstruction(2) > 0.5f ? 1.0f : 0.0f;
			float b3 = rsa.getVisibleNodeReconstruction(3) > 0.5f ? 1.0f : 0.0f;

			rsa.activate(sparsity, dutyCycleDecay);

			std::cout << "-----------------------------------" << std::endl;

			std::cout << (sequence[j][0]) << std::endl;
			std::cout << (sequence[j][1]) << std::endl;
			std::cout << (sequence[j][2]) << std::endl;
			std::cout << (sequence[j][3]) << std::endl;

			std::cout << b0 << std::endl;
			std::cout << b1 << std::endl;
			std::cout << b2 << std::endl;
			std::cout << b3 << std::endl;

			std::cout << std::endl;

			correctCount += 4.0f - (std::abs(sequence[j][0] - b0) +
				std::abs(sequence[j][1] - b1) +
				std::abs(sequence[j][2] - b2) +
				std::abs(sequence[j][3] - b3));

			totalCount += 4.0f;
		}
	}

	std::cout << "% correct: " << (correctCount / totalCount) * 100.0f << std::endl;

	system("pause");

	return 0;
}*/

/*struct BufferAndLabel {
	sf::SoundBuffer _buffer;
	int _classLabel;
	std::vector<mfcc::AudioFeatureMFCC> _features;
	std::vector<float> _sdr;
};

int main() {
	std::mt19937 generator(time(nullptr));

	// ------------------------------ Learner Initialization ------------------------------

	int numHidden = 80;
	float sparsity = 8.01f / numHidden;
	float dutyCycleDecay = 0.01f;

	deep::RecurrentSparseAutoencoder rsa;

	rsa.createRandom(26, numHidden, sparsity, -0.1f, 0.1f, generator);

	std::uniform_real_distribution<float> weightDist(-0.1f, 0.1f);

	int classes = 2;

	nn::FeedForwardNeuralNetwork ffnn;

	ffnn.createRandom(numHidden, classes, 1, 40, -0.01f, 0.01f, generator);

	// ---------------------------------- Sound Loading ----------------------------------

	std::vector<BufferAndLabel> testSounds(44);

	int maxInputSize = 0;

	for (int i = 0; i < testSounds.size(); i++) {
		std::string fn = "Resources/testSound" + std::to_string(i + 1) + ".wav";

		testSounds[i]._buffer.loadFromFile(fn);

		testSounds[i]._classLabel = i % 2;
	}

	int soundCount = testSounds.size() + 1;

	// --------------------------------- Extract Features --------------------------------

	int featureSamplesLength = 400;
	int featureSamplesStep = 200;

	mfcc::MelFilterBank bank;
	bank.create(26, featureSamplesLength, 300.0f, 8000.0f, 8000.0f);

	for (int t = 0; t < testSounds.size(); t++) {
		int numFeatures = (testSounds[t]._buffer.getSampleCount() - featureSamplesLength + 1) / featureSamplesStep;

		int numSamplesUse = featureSamplesStep * (numFeatures + 1) + featureSamplesLength;

		std::vector<short> samples(numSamplesUse, 0);

		for (int s = 0; s < testSounds[t]._buffer.getSampleCount(); s++)
			samples[s] = testSounds[t]._buffer.getSamples()[s];

		testSounds[t]._features.resize(numFeatures);
		
		int start = 0;

		for (int f = 0; f < numFeatures; f++) {
			testSounds[t]._features[f].extract(samples, start, featureSamplesLength, bank);

			start += featureSamplesStep;
		}

		std::cout << "Features extracted from sound " << t << std::endl;
	}

	// ----------------------------------- Training -----------------------------------

	std::uniform_int_distribution<int> sampleDist(0, testSounds.size() - 1);

	for (int t = 0; t < 500; t++) {
		int i = sampleDist(generator);

		rsa.clearMemory();

		for (int f = 0; f < testSounds[i]._features.size(); f++) {
			rsa.stepBegin();

			for (int s = 0; s < 26; s++)
				rsa.setVisibleNodeState(s, testSounds[i]._features[f].getCoeff(s) * 0.01f);

			rsa.learn(sparsity, 0.0f, 0.2f, 0.1f, 0.0f, 1.0f, 0.5f, 1.0f, 1.0f);

			rsa.activate(sparsity, dutyCycleDecay);
		}

		if (t % 10 == 0) {
			std::cout << "Unsupervised: " << t << std::endl;

			std::cout << "SDR: " << std::endl;

			for (int p = 0; p < numHidden; p++) {
				if (rsa.getHiddenNodeState(p) > 0.001f)
					std::cout << "X";
				else
					std::cout << "_";
			}
		}
	}

	for (int t = 0; t < testSounds.size(); t++) {
		rsa.clearMemory();

		for (int f = 0; f < testSounds[t]._features.size(); f++) {
			rsa.stepBegin();

			for (int s = 0; s < 26; s++)
				rsa.setVisibleNodeState(s, testSounds[t]._features[f].getCoeff(s) * 0.01f);

			rsa.activate(sparsity, dutyCycleDecay);
		}

		testSounds[t]._sdr.resize(numHidden);

		for (int i = 0; i < numHidden; i++)
			testSounds[t]._sdr[i] = rsa.getHiddenNodeState(i);
	}

	for (int t = 0; t < 500000; t++) {
		int i = sampleDist(generator);

		// Classifier
		for (int p = 0; p < numHidden; p++) {
			ffnn.setInput(p, testSounds[i]._sdr[p]);
		}

		ffnn.activate();

		int maxIndex = 0;

		for (int c = 1; c < classes; c++) {
			if (ffnn.getOutput(c) > ffnn.getOutput(maxIndex))
				maxIndex = c;
		}

		std::vector<float> targets(classes, 0.0f);

		targets[testSounds[i]._classLabel] = 1.0f;

		nn::FeedForwardNeuralNetwork::Gradient grad;

		ffnn.getGradient(targets, grad);

		ffnn.moveAlongGradientMomentum(grad, 0.004f, 0.3f);

		ffnn.decayWeights(0.0001f);

		if (t % 1000 == 0)
			std::cout << "Training: " << t << " Prediction: " << maxIndex << " Actual: " << testSounds[i]._classLabel << std::endl;
	}

	// ----------------------------------- Windowing ----------------------------------

	if (!sf::SoundBufferRecorder::isAvailable())
		std::cerr << "No recording device available!" << std::endl;

	sf::RenderWindow window;

	sf::SoundBufferRecorder recorder;

	sf::Clock clock;

	window.create(sf::VideoMode(800, 600), "Speech Recognition", sf::Style::Default);

	float dt = 0.017f;

	bool quit = false;

	bool recording = false;

	do {
		clock.restart();

		// ----------------------------- Input -----------------------------

		sf::Event windowEvent;

		while (window.pollEvent(windowEvent))
		{
			switch (windowEvent.type)
			{
			case sf::Event::Closed:
				quit = true;
				break;

			case sf::Event::KeyReleased:
				if (windowEvent.key.code == sf::Keyboard::R) {
					if (recording) {
						recorder.stop();

						const sf::SoundBuffer &buffer = recorder.getBuffer();

						buffer.saveToFile("testSound" + std::to_string(soundCount) + ".wav");

						soundCount++;

						sf::Sound sound;

						sound.setBuffer(buffer);

						sound.play();

						int numFeatures = (buffer.getSampleCount() - featureSamplesLength + 1) / featureSamplesStep;

						int numSamplesUse = featureSamplesStep * (numFeatures + 1) + featureSamplesLength;

						std::vector<mfcc::AudioFeatureMFCC> features(numFeatures);

						std::vector<short> samples(numSamplesUse, 0);

						for (int s = 0; s < buffer.getSampleCount(); s++)
							samples[s] = buffer.getSamples()[s];

						features.resize(numFeatures);

						int start = 0;

						for (int f = 0; f < numFeatures; f++) {
							features[f].extract(samples, start, featureSamplesLength, bank);

							start += featureSamplesStep;
						}

						// Classify recording
						rsa.clearMemory();

						for (int f = 0; f < features.size(); f++) {
							rsa.stepBegin();

							for (int s = 0; s < 26; s++)
								rsa.setVisibleNodeState(s, features[f].getCoeff(s) * 0.01f);

							rsa.activate(sparsity, dutyCycleDecay);
						}

						// Classifier
						std::cout << "SDR: " << std::endl;

						for (int p = 0; p < numHidden; p++) {
							if (rsa.getHiddenNodeState(p) > 0.001f)
								std::cout << "X";
							else
								std::cout << "_";

							ffnn.setInput(p, rsa.getHiddenNodeState(p));
						}

						ffnn.activate();

						int maxIndex = 0;

						for (int c = 0; c < classes; c++) {
							if (ffnn.getOutput(c) > ffnn.getOutput(maxIndex))
								maxIndex = c;
						}

						std::cout << "Class: " << maxIndex << std::endl;

						recording = false;
					}
					else {
						recorder.start();

						recording = true;
					}
				}
			}
		}

		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			quit = true;

		window.display();

		float dt = clock.getElapsedTime().asSeconds();
	} while (!quit);

	return 0;
}*/

/*int main() {
	std::mt19937 generator(time(nullptr));

	text::Word2SDR w2sdr;
	text::Word2SDR::Settings settings;

	w2sdr.createRandom(64, settings, -0.1f, 0.1f, 0.5f, generator);

	for (int i = 0; i < 10; i++) {
		w2sdr.clearMemory();

		std::ifstream fromFile("Resources/corpus.txt");

		w2sdr.read(fromFile);

		std::cout << "Corpus Iteration " << i << std::endl;
	}

	w2sdr.clearMemory();

	std::ifstream fromFile("Resources/corpus.txt");

	std::string firstWord;
	fromFile >> firstWord;

	w2sdr.show(firstWord);

	std::cout << firstWord << " ";

	for (int i = 0; i < 200; i++) {
		std::cout << w2sdr.getPrediction() << " ";

		w2sdr.show(w2sdr.getPrediction());
	}

	w2sdr.clearMemory();

	const std::string escapeWord = "!EXIT";
	const std::string clearMemoryWord = "!MEM";

	std::cout << "Ready" << std::endl;

	std::string word = "";

	while (word != escapeWord) {
		std::cin >> word;

		if (word == clearMemoryWord)
			w2sdr.clearMemory();
		else {
			w2sdr.show(word);

			std::cout << "Prediction: " << w2sdr.getPrediction() << std::endl;
		}
	}

	system("pause");

	return 0;
}*/

float sigmoid(float x) {
	return 1.0f / (1.0f + std::exp(-x));
}

int main() {
	std::mt19937 generator(time(nullptr));

	sf::Image img;

	img.loadFromFile("testImage.png");

	deep::SparseCoder sc;

	int winWidth = 16;
	int winHeight = 16;
	int sdrWidth = 18;
	int sdrHeight = 18;

	float sparsity = 12.0f / (sdrWidth * sdrHeight);

	sc.createRandom(winWidth * winHeight, sdrWidth * sdrHeight, generator);

	std::uniform_int_distribution<int> widthDist(0, img.getSize().x - winWidth);
	std::uniform_int_distribution<int> heightDist(0, img.getSize().y - winHeight);

	sf::RenderWindow window;

	sf::SoundBufferRecorder recorder;

	sf::Clock clock;

	window.create(sf::VideoMode(800, 800), "SDRs", sf::Style::Default);

	for (int iter = 0; iter < 100000; iter++) {
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
			break;

		int wx = widthDist(generator);
		int wy = heightDist(generator);

		int inputIndex = 0;

		for (int x = 0; x < winWidth; x++)
			for (int y = 0; y < winHeight; y++) {
				int tx = wx + x;
				int ty = wy + y;

				sc.setVisibleInput(inputIndex++, img.getPixel(tx, ty).r / 255.0f);
			}

		for (int i = 0; i < 40; i++) {
			sc.activate(sparsity, 20.0f, 0.05f);

			sc.stepEnd();
		}

		sc.reconstruct();

		sc.learn(0.02f, 0.01f, 0.02f, sparsity);

		if (iter % 10 == 0) {
			std::cout << "Iteration " << iter << std::endl;

			float minWeight = 9999.0f;
			float maxWeight = -9999.0f;

			for (int sx = 0; sx < sdrWidth; sx++)
				for (int sy = 0; sy < sdrHeight; sy++) {
					for (int x = 0; x < winWidth; x++)
						for (int y = 0; y < winHeight; y++) {
							float w = sc.getVHWeight(sx + sy * sdrWidth, x + y * winWidth);

							minWeight = std::min(minWeight, w);
							maxWeight = std::max(maxWeight, w);
						}
				}

			sf::Image rfs;
			rfs.create(sdrWidth * winWidth, sdrHeight * winHeight);

			float scalar = 1.0f / (maxWeight - minWeight);

			for (int sx = 0; sx < sdrWidth; sx++)
				for (int sy = 0; sy < sdrHeight; sy++) {
					for (int x = 0; x < winWidth; x++)
						for (int y = 0; y < winHeight; y++) {
							sf::Color color;

							color.r = color.b = color.g = 255 * scalar * (sc.getVHWeight(sx + sy * sdrWidth, x + y * winWidth) - minWeight);
							color.a = 255;

							rfs.setPixel(sx * winWidth + x, sy * winHeight + y, color);
						}
				}

			sf::Texture t;
			t.loadFromImage(rfs);

			sf::Sprite s;
			s.setTexture(t);

			s.setScale(2.0f, 2.0f);

			window.clear();
			window.draw(s);

			for (int sx = 0; sx < sdrWidth; sx++)
				for (int sy = 0; sy < sdrHeight; sy++) {
					if (sc.getHiddenState(sx + sy * sdrWidth) > 0.0f) {
						sf::RectangleShape rs;

						rs.setPosition(sx * winWidth * 2.0f, sy * winHeight * 2.0f);
						rs.setOutlineColor(sf::Color::Red);
						rs.setFillColor(sf::Color::Transparent);
						rs.setOutlineThickness(2.0f);

						rs.setSize(sf::Vector2f(winWidth * 2.0f, winHeight * 2.0f));

						window.draw(rs);
					}
				}

			window.display();
		}
	}

	float minWeight = 9999.0f;
	float maxWeight = -9999.0f;

	for (int sx = 0; sx < sdrWidth; sx++)
		for (int sy = 0; sy < sdrHeight; sy++) {
			for (int x = 0; x < winWidth; x++)
				for (int y = 0; y < winHeight; y++) {
					float w = sc.getVHWeight(sx + sy * sdrWidth, x + y * winWidth);
					
					minWeight = std::min(minWeight, w);
					maxWeight = std::max(maxWeight, w);
				}
		}

	sf::Image rfs;
	rfs.create(sdrWidth * winWidth, sdrHeight * winHeight);

	float scalar = 1.0f / (maxWeight - minWeight);

	for (int sx = 0; sx < sdrWidth; sx++)
		for (int sy = 0; sy < sdrHeight; sy++) {
			for (int x = 0; x < winWidth; x++)
				for (int y = 0; y < winHeight; y++) {
					sf::Color color;

					color.r = color.b = color.g = 255 * scalar * (sc.getVHWeight(sx + sy * sdrWidth, x + y * winWidth) - minWeight);
					color.a = 255;

					rfs.setPixel(sx * winWidth + x, sy * winHeight + y, color);
				}
		}

	rfs.saveToFile("rfs.png");

	return 0;
}