//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include "cuda_runtime.h"	
#include "device_launch_parameters.h"

#include "ped_agent.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, TASK, MOVESEQ};

	class Model
	{
	public:

		// Sets everything up
		void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
		
		// Coordinates a time step in the scenario: move all agents by one step (if applicable).
		void tick();

		// Returns the agents of this scenario
		const std::vector<Tagent*> getAgents() const { return agents; };

		// Returns the destinations of this scenario
		const std::vector<Twaypoint*> getDest() const { return destinations; };



		// Returns the the X vector of this scenario
		float *getVecX() const { return X; };

		// Returns the the Y vector of this scenario
		float *getVecY() const { return Y; };

		// Returns the the X vector of this scenario
		float *getVecXdes() const { return destX; };

		// Returns the the Y vector of this scenario
		float *getVecYdes() const { return destY; };

		// Adds an agent to the tree structure
		void placeAgent(const Ped::Tagent *a);

		// Cleans up the tree and restructures it. Worth calling every now and then.
		void cleanup();
		~Model();

		// Returns the heatmap visualizing the density of agents
		int const * const * getHeatmap() const { return blurred_heatmap; };
		int getHeatmapSize() const;

	private:

		// Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;

		// The agents in this scenario
		std::vector<Tagent*> agents;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		int region1, region2, region3, region4;

		// Vector of x and y coordinates in this scenario
		float *X;
		float *Y;
		// Vector of x and y coordinates in this scenario
		float *destX;
		float *destY;
		float *destR;

		std::set<Ped::Tagent*> region1list;
		std::set<Ped::Tagent*> region2list;
		std::set<Ped::Tagent*> region3list;
		std::set<Ped::Tagent*> region4list;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent);
		void movecrit(Ped::Tagent *agent);

		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////
		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist) const;
		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE
#define WEIGHTSUM 273

		// The heatmap representing the density of agents
		int ** heatmap;
		int * d_heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;
		int * d_scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;
		int * d_blurred_heatmap;

		// Desired positions of agents
		int *desiredX;
		int *desiredY;
		int *d_desiredX;
		int *d_desiredY;

		// Number of agents in scenario
		int agentsSize;

		void setupHeatmapCuda();
		void setupHeatmapSeq();
		void updateHeatmapSeq();
		void updateHeatmapCuda();
	
	};
}
#endif
