#include "stdafx.h"
#include "FeatureMap.h"
#include "MatrixOperations.h"


FeatureMap::FeatureMap(int width,int height,int outp_w,int outp_h)
{ w=width;
  h=height;
  output_w = outp_w;
  output_h = outp_h;
  b=0;
}

FeatureMap::~FeatureMap(void)
{}
void FeatureMap::get_output()
{}
void FeatureMap::correct_weights()
{}
