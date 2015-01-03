#include <SFML/Graphics.hpp>

#include <rbf/SDRRBFNetwork.h>

#include <time.h>
#include <iostream>
#include <random>
#include <array>
#include <fstream>
#include <sstream>

#include <dirent.h>

int main() {
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir("c:\\src\\")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			printf("%s\n", ent->d_name);
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}

	rbf::SDRRBFNetwork sdrrbfnet;

	std::mt19937 generator(time(nullptr));

	std::vector<rbf::SDRRBFNetwork::LayerDesc> layerDescs(4);

	layerDescs[0]._rbfWidth = 60;
	layerDescs[0]._rbfHeight = 60;
	layerDescs[0]._receptiveRadius = 2;
	layerDescs[0]._inhibitionRadius = 3;
	layerDescs[0]._outputMultiplier = 0.4f;

	layerDescs[1]._rbfWidth = 50;
	layerDescs[1]._rbfHeight = 50;
	layerDescs[1]._receptiveRadius = 2;
	layerDescs[1]._inhibitionRadius = 3;
	layerDescs[1]._outputMultiplier = 0.6f;

	layerDescs[2]._rbfWidth = 40;
	layerDescs[2]._rbfHeight = 40;
	layerDescs[2]._receptiveRadius = 2;
	layerDescs[2]._inhibitionRadius = 3;
	layerDescs[2]._outputMultiplier = 0.8f;

	layerDescs[3]._rbfWidth = 30;
	layerDescs[3]._rbfHeight = 30;
	layerDescs[3]._receptiveRadius = 2;
	layerDescs[3]._inhibitionRadius = 3;
	layerDescs[3]._outputMultiplier = 1.0f;

	sdrrbfnet.createRandom(28, 28, layerDescs, 10, 0.05f, 0.4f, 0.1f, 0.3f, -0.02f, 0.02f, generator);


}