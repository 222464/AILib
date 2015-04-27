/*#include <SFML/Graphics.hpp>

#include <rbf/SDRNetwork.h>

#include <time.h>
#include <iostream>
#include <random>
#include <array>
#include <fstream>
#include <sstream>
#include <string>

#include <dirent.h>

struct Label {
	std::string _name;

	std::vector<std::string> _members;

	Label(const std::string &name)
		: _name(name)
	{}
};

int main() {
	std::vector<Label> labels;

	DIR *dir;

	dirent *ent;

	int totalImages = 0;

	int numReadFromEachDir = 5;

	if ((dir = opendir("Resources/train")) != nullptr) {
		readdir(dir);
		readdir(dir);

		while ((ent = readdir(dir)) != nullptr) {
			labels.push_back(Label(ent->d_name));

			DIR *innerDir;

			// Get names of all files in this directory as members
			if ((innerDir = opendir((std::string("Resources/train/") + std::string(ent->d_name)).c_str())) != nullptr) {
				dirent *innerEnt;

				readdir(innerDir);
				readdir(innerDir);

				while ((innerEnt = readdir(innerDir)) != nullptr && labels.back()._members.size() < numReadFromEachDir) {
					if (std::string(innerEnt->d_name) != "") {
						labels.back()._members.push_back(innerEnt->d_name);
						totalImages++;
					}
				}

				closedir(innerDir);
			}
		}

		closedir(dir);
	}
	else {
		std::cerr << "Could not open Resources/train!" << std::endl;

		return 0;
	}

	sdr::SDRNetwork sdrnet;

	std::mt19937 generator(time(nullptr));

	std::vector<sdr::SDRNetwork::LayerDesc> layerDescs(3);

	layerDescs[0]._width = 64;
	layerDescs[0]._height = 64;
	layerDescs[0]._receptiveRadius = 9;
	layerDescs[0]._inhibitionRadius = 6;

	layerDescs[1]._width = 48;
	layerDescs[1]._height = 48;
	layerDescs[1]._receptiveRadius = 9;
	layerDescs[1]._inhibitionRadius = 6;

	layerDescs[2]._width = 32;
	layerDescs[2]._height = 32;
	layerDescs[2]._receptiveRadius = 9;
	layerDescs[2]._inhibitionRadius = 6;

	const int inputWidth = 64;
	const int inputHeight = 64;

	sdrnet.createRandom(inputWidth, inputHeight, layerDescs, labels.size(), -0.01f, 0.01f, 0.0f, 0.05f, -0.01f, 0.01f, generator);

	sf::RenderTexture resizeTexture;
	resizeTexture.create(inputWidth, inputHeight);

	sf::RectangleShape clearRect;
	clearRect.setSize(sf::Vector2f(inputWidth, inputHeight));

	clearRect.setFillColor(sf::Color::White);

	std::uniform_int_distribution<int> distImage(0, totalImages - 1);

	const float byteInv = 1.0f / 255.0f;

	int unsupervisedIterations = 1500;
	int supervisedIterations = 1000;

	for (int i = 0; i < unsupervisedIterations; i++) {
		resizeTexture.draw(clearRect);

		int index = distImage(generator);

		int li = 0;
		int mi = 0;
		int c = 0;

		for (int l = 0; l < labels.size(); l++) {
			int pc = c;
			c += labels[l]._members.size();

			if (c > index) {
				li = l;
				mi = index - pc;
				break;
			}
		}

		sf::Texture source;
		if (!source.loadFromFile("Resources/train/" + labels[li]._name + "/" + labels[li]._members[mi])) {
			std::cerr << "Could not load \"" << ("Resources/train/" + labels[li]._name + "/" + labels[li]._members[mi]) << "\"!" << std::endl;

			continue;
		}

		source.setSmooth(true);

		sf::Sprite sprite;
		sprite.setTexture(source);

		sprite.setOrigin(source.getSize().x * 0.5f, source.getSize().y * 0.5f);

		sprite.setPosition(inputWidth / 2.0f, inputHeight / 2.0f);

		float scale = std::min(static_cast<float>(inputWidth) / source.getSize().x, static_cast<float>(inputHeight) / source.getSize().y);

		sprite.setScale(scale, scale);

		resizeTexture.draw(sprite);

		resizeTexture.display();

		sf::Image resizedImage = resizeTexture.getTexture().copyToImage();

		std::vector<float> input(resizedImage.getSize().x * resizedImage.getSize().y);

		for (int x = 0; x < inputWidth; x++)
		for (int y = 0; y < inputHeight; y++) {
			sf::Color color = resizedImage.getPixel(x, y);

			float greyscale = color.r * byteInv * 0.299f + color.g * byteInv * 0.587f + color.b * byteInv * 0.114f;

			input[x + y * inputWidth] = greyscale;
		}

		std::vector<float> output;

		sdrnet.getOutput(input, output, generator);

		sdrnet.updateUnsupervised(input, 0.001f, 0.01f, 0.005f);

		if (i % 25 == 0)
			std::cout << i / static_cast<float>(unsupervisedIterations) * 100.0f << "%" << std::endl;
	}

	sf::Image rfImage;

	sdrnet.getReceptiveFields(0, rfImage);

	rfImage.saveToFile("rfsnet.png");

	for (int i = 0; i < supervisedIterations; i++) {
		resizeTexture.draw(clearRect);

		int index = distImage(generator);

		int li = 0;
		int mi = 0;
		int c = 0;

		for (int l = 0; l < labels.size(); l++) {
			int pc = c;
			c += labels[l]._members.size();

			if (c > index) {
				li = l;
				mi = index - pc;
				break;
			}
		}

		sf::Texture source;
		if (!source.loadFromFile("Resources/train/" + labels[li]._name + "/" + labels[li]._members[mi])) {
			std::cerr << "Could not load \"" << ("Resources/train/" + labels[li]._name + "/" + labels[li]._members[mi]) << "\"!" << std::endl;

			continue;
		}

		source.setSmooth(true);

		sf::Sprite sprite;
		sprite.setTexture(source);

		sprite.setOrigin(source.getSize().x * 0.5f, source.getSize().y * 0.5f);

		sprite.setPosition(inputWidth / 2.0f, inputHeight / 2.0f);

		float scale = std::min(static_cast<float>(inputWidth) / source.getSize().x, static_cast<float>(inputHeight) / source.getSize().y);

		sprite.setScale(scale, scale);

		resizeTexture.draw(sprite);

		resizeTexture.display();

		sf::Image resizedImage = resizeTexture.getTexture().copyToImage();

		std::vector<float> input(resizedImage.getSize().x * resizedImage.getSize().y);

		for (int x = 0; x < inputWidth; x++)
		for (int y = 0; y < inputHeight; y++) {
			sf::Color color = resizedImage.getPixel(x, y);

			float greyscale = color.r * byteInv * 0.299f + color.g * byteInv * 0.587f + color.b * byteInv * 0.114f;

			input[x + y * inputWidth] = greyscale;
		}

		std::vector<float> output;

		sdrnet.getOutput(input, output, generator);

		int givenLabel = 0;

		for (int l = 1; l < output.size(); l++)	 {
			if (output[l] > output[givenLabel])
				givenLabel = l;
		}

		std::vector<float> target(output.size(), 0.0f);

		target[li] = 1.0f;

		if (li == givenLabel)
			std::cout << "g";

		sdrnet.updateSupervised(input, output, target, 0.005f, 0.001f, 0.3f);

		if (i % 25 == 0)
			std::cout << i / static_cast<float>(supervisedIterations) * 100.0f << "%" << std::endl;
	}

	std::ofstream toFile("result.csv");

	toFile << "image,";

	for (int l = 0; l < labels.size(); l++)
	if (l == labels.size() - 1)
		toFile << labels[l]._name;
	else
		toFile << labels[l]._name << ",";

	toFile << std::endl;

	// Export results
	DIR *testDir;

	dirent *testEnt;

	if ((testDir = opendir("Resources/test/test")) != nullptr) {
		readdir(testDir);
		readdir(testDir);

		while ((testEnt = readdir(testDir)) != nullptr) {
			std::string fullName = "Resources/test/test/" + std::string(testEnt->d_name);

			sf::Texture source;
			if (!source.loadFromFile(fullName)) {
				std::cerr << "Could not load \"" << fullName << "\"!" << std::endl;

				continue;
			}

			std::cout << "Eval: " << std::string(testEnt->d_name) << std::endl;

			toFile << std::string(testEnt->d_name) + ",";

			source.setSmooth(true);

			sf::Sprite sprite;
			sprite.setTexture(source);

			sprite.setOrigin(source.getSize().x * 0.5f, source.getSize().y * 0.5f);

			sprite.setPosition(inputWidth / 2.0f, inputHeight / 2.0f);

			float scale = std::min(static_cast<float>(inputWidth) / source.getSize().x, static_cast<float>(inputHeight) / source.getSize().y);

			sprite.setScale(scale, scale);

			resizeTexture.draw(sprite);

			resizeTexture.display();

			sf::Image resizedImage = resizeTexture.getTexture().copyToImage();

			std::vector<float> input(resizedImage.getSize().x * resizedImage.getSize().y);

			for (int x = 0; x < inputWidth; x++)
			for (int y = 0; y < inputHeight; y++) {
				sf::Color color = resizedImage.getPixel(x, y);

				float greyscale = color.r * byteInv * 0.299f + color.g * byteInv * 0.587f + color.b * byteInv * 0.114f;

				input[x + y * inputWidth] = greyscale;
			}

			std::vector<float> output;

			sdrnet.getOutput(input, output, generator);

			for (int i = 0; i < output.size(); i++)
			if (i == output.size() - 1)
				toFile << output[i];
			else
				toFile << output[i] << ",";

			toFile << std::endl;
		}

		closedir(testDir);
	}

	toFile.close();

	sf::RenderWindow window;

	window.create(sf::VideoMode(800, 600), "SDRRBFNetwork", sf::Style::Default);

	// Test
	int correct = 0;
	int testTotal = 0;
	float logSum = 0.0f;

	int currentLi = 0;
	int currentMi = 0;
	int currentGiven = 0;

	bool testKeyPressedLastFrame = false;

	// ----------------------------- Game Loop -----------------------------

	bool quit = false;

	sf::Clock clock;

	float dt = 0.017f;

	bool first = true;

	sf::Font font;
	font.loadFromFile("Resources/arial.ttf");

	sf::Image currentImage;

	std::vector<sf::Image> rbfImages;

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

		if (first || (!testKeyPressedLastFrame && sf::Keyboard::isKeyPressed(sf::Keyboard::T))) {
			resizeTexture.draw(clearRect);

			int index = distImage(generator);

			int li = 0;
			int mi = 0;
			int c = 0;

			for (int l = 0; l < labels.size(); l++) {
				int pc = c;
				c += labels[l]._members.size();

				if (c > index) {
					li = l;
					mi = index - pc;
					break;
				}
			}

			sf::Texture source;
			if (!source.loadFromFile("Resources/train/" + labels[li]._name + "/" + labels[li]._members[mi])) {
				std::cerr << "Could not load \"" << ("Resources/train/" + labels[li]._name + "/" + labels[li]._members[mi]) << "\"!" << std::endl;

				continue;
			}

			source.setSmooth(true);

			sf::Sprite sprite;
			sprite.setTexture(source);

			sprite.setOrigin(source.getSize().x * 0.5f, source.getSize().y * 0.5f);

			sprite.setPosition(inputWidth / 2.0f, inputHeight / 2.0f);

			float scale = std::min(static_cast<float>(inputWidth) / source.getSize().x, static_cast<float>(inputHeight) / source.getSize().y);

			sprite.setScale(scale, scale);

			resizeTexture.draw(sprite);

			resizeTexture.display();

			sf::Image resizedImage = resizeTexture.getTexture().copyToImage();

			currentImage = resizedImage;

			std::vector<float> input(resizedImage.getSize().x * resizedImage.getSize().y);

			for (int x = 0; x < inputWidth; x++)
			for (int y = 0; y < inputHeight; y++) {
				sf::Color color = resizedImage.getPixel(x, y);

				float greyscale = color.r * byteInv * 0.299f + color.g * byteInv * 0.587f + color.b * byteInv * 0.114f;

				input[x + y * inputWidth] = greyscale;
			}

			std::vector<float> output;

			sdrnet.getOutput(input, output, generator);

			int givenLabel = 0;

			for (int l = 1; l < output.size(); l++)
			if (output[l] > output[givenLabel])
				givenLabel = l;

			if (givenLabel == li)
				correct++;
			
			const float threshold = std::numeric_limits<float>::epsilon();

			logSum += std::log(std::max(threshold, std::min(1.0f - threshold, output[li])));

			testTotal++;

			currentLi = li;
			currentMi = mi;
			currentGiven = givenLabel;

			sdrnet.getImages(rbfImages);
		}

		first = false;

		testKeyPressedLastFrame = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

		// ---------------------------- Rendering ----------------------------

		window.clear(sf::Color::Blue);

		sf::Texture currentTexture;
		currentTexture.loadFromImage(currentImage);

		sf::Sprite sprite;
		sprite.setTexture(currentTexture);

		sprite.setOrigin(inputWidth / 2, inputHeight / 2);

		sprite.setPosition(400, 300);

		sprite.setScale(2.0f, 2.0f);

		window.draw(sprite);

		sf::Text text;
		text.setFont(font);

		text.setString("Actual: " + labels[currentLi]._name);

		text.setPosition(2.0f, 40.0f);

		window.draw(text);

		text.setString("Guess: " + labels[currentGiven]._name);

		text.setPosition(2.0f, 2.0f);

		window.draw(text);

		text.setString("% Correct: " + std::to_string(static_cast<float>(correct) / testTotal * 100.0f));

		text.setPosition(2.0f, 548.0f);

		window.draw(text);

		text.setString("Log Loss: " + std::to_string(-logSum / testTotal));

		text.setPosition(2.0f, 508.0f);

		window.draw(text);

		// Render RBF node states
		for (int l = 0; l < rbfImages.size(); l++) {
			sf::Texture layerTexture;
			layerTexture.loadFromImage(rbfImages[l]);

			sf::Sprite layerSprite;
			layerSprite.setTexture(layerTexture);

			layerSprite.setScale(2.0f, 2.0f);

			layerSprite.setPosition(800.0f - layerTexture.getSize().x * 2.0f, l * 128.0f);

			window.draw(layerSprite);
		}

		// -------------------------------------------------------------------

		window.display();

		dt = clock.getElapsedTime().asSeconds();
	} while (!quit);
}*/