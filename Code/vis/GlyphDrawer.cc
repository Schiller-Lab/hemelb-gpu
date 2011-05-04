#include "vis/GlyphDrawer.h"
#include "vis/Control.h"

namespace hemelb
{
  namespace vis
  {
    // Constructor
    GlyphDrawer::GlyphDrawer(geometry::LatticeData* iLatDat,
                             Screen* iScreen,
                             DomainStats* iDomainStats,
                             Viewpoint* iViewpoint,
                             VisSettings* iVisSettings) :
      mLatDat(iLatDat), mScreen(iScreen), mDomainStats(iDomainStats), mViewpoint(iViewpoint),
          mVisSettings(iVisSettings)
    {
      int n = -1;

      // Iterate over the first site in each block.
      for (site_t i = 0; i < mLatDat->GetXSiteCount(); i += mLatDat->GetBlockSize())
      {
        for (site_t j = 0; j < mLatDat->GetYSiteCount(); j += mLatDat->GetBlockSize())
        {
          for (site_t k = 0; k < mLatDat->GetZSiteCount(); k += mLatDat->GetBlockSize())
          {
            ++n;

            // Get the block data for this block - if it has no site data, move on.
            geometry::LatticeData::BlockData * map_block_p = mLatDat->GetBlock(n);

            if (map_block_p->site_data == NULL)
            {
              continue;
            }

            // We put the glyph at the site at the centre of the block...
            const site_t site_i = (mLatDat->GetBlockSize() >> 1);
            const site_t site_j = (mLatDat->GetBlockSize() >> 1);
            const site_t site_k = (mLatDat->GetBlockSize() >> 1);

            const site_t siteIdOnBlock = ( ( (site_i << mLatDat->GetLog2BlockSize()) + site_j)
                << mLatDat->GetLog2BlockSize()) + site_k;

            // ... (only if there's fluid there).
            if (map_block_p->site_data[siteIdOnBlock] & BIG_NUMBER3)
            {
              continue;
            }

            // Create a glyph at the desired location
            Glyph *lGlyph = new Glyph();

            lGlyph->x = float (i + site_i) - 0.5F * float (mLatDat->GetXSiteCount());
            lGlyph->y = float (j + site_j) - 0.5F * float (mLatDat->GetYSiteCount());
            lGlyph->z = float (k + site_k) - 0.5F * float (mLatDat->GetZSiteCount());

            lGlyph->f = mLatDat->GetFOld(map_block_p->site_data[siteIdOnBlock] * D3Q15::NUMVECTORS);

            mGlyphs.push_back(lGlyph);
          } // for k
        } // for j
      } // for i

    }

    // Destructor
    GlyphDrawer::~GlyphDrawer()
    {
      for (size_t ii = 0; ii < mGlyphs.size(); ii++)
      {
        delete mGlyphs[ii];
      }
    }

    /**
     * Perform the rendering for each glyph.
     */
    void GlyphDrawer::Render()
    {
      // For each glyph...
      for (site_t n = 0; n < (site_t) mGlyphs.size(); n++)
      {
        // ... get the density and velocity at that point...
        distribn_t density;
        distribn_t vx, vy, vz;
        D3Q15::CalculateDensityAndVelocity(mGlyphs[n]->f, density, vx, vy, vz);

        // ... calculate the velocity vector multiplier...
        const double temp = mVisSettings->glyphLength * ((distribn_t) mLatDat->GetBlockSize())
            * mDomainStats->velocity_threshold_max_inv / density;

        // ... calculate the two ends of the line we're going to draw...
        float p1[3], p2[3];
        p1[0] = mGlyphs[n]->x;
        p1[1] = mGlyphs[n]->y;
        p1[2] = mGlyphs[n]->z;

        p2[0] = mGlyphs[n]->x + (float) vx * temp;
        p2[1] = mGlyphs[n]->y + (float) vy * temp;
        p2[2] = mGlyphs[n]->z + (float) vz * temp;

        // ... transform to the location on the screen, and render.
        float p3[3], p4[3];
        mViewpoint->Project(p1, p3);
        mViewpoint->Project(p2, p4);

        mScreen->Transform<float> (p3, p3);
        mScreen->Transform<float> (p4, p4);

        mScreen->RenderLine(p3, p4, mVisSettings);
      }
    }

  }
}
