#ifndef HEMELB_VIS_BLOCKTRAVERSERWITHVISITEDBLOCKTRACKER_H
#define HEMELB_VIS_BLOCKTRAVERSERWITHVISITEDBLOCKTRACKER_H

#include "geometry/SiteTraverser.h"
#include "geometry/BlockTraverser.h"

namespace hemelb
{
  namespace vis
  {
    namespace raytracer 
    {
      
      //On top of BlockTraverser, this class also contains 
      //a record of which blocks have been visited, which
      //is neccessary for the algoritm which uses this. No locations are automatically
      //marked visited, and methods have been created to assist with random access
      //of the lattice data as required by the algorithm
      class BlockTraverserWithVisitedBlockTracker
	: public geometry::BlockTraverser
      {
      public:
	BlockTraverserWithVisitedBlockTracker
	  (const geometry::LatticeData& iLatDat);

	~BlockTraverserWithVisitedBlockTracker();

	//Tranverses the block until the next unvisited block is reached.
	//Returns false if the end of the Volume is reached
	bool GoToNextUnvisitedBlock();
	
	bool IsCurrentBlockVisited();

	bool IsBlockVisited(site_t n);
	bool IsBlockVisited(util::Vector3D<site_t> n);

	void MarkCurrentBlockVisited();

	void MarkBlockVisited(site_t n);
	void MarkBlockVisited(util::Vector3D<site_t> location);

      private:
	bool* mBlockVisited;
      }; 
    }
  }
}


#endif // HEMELB_VIS_BLOCKTRAVERSERWITHVISITEDBLOCKTRACKER_H
