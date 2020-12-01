#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "theia/sfm/pose/perspective_three_point.h"
#include "theia/util/python_types.h"

#include <glog/logging.h>
#include <math.h>
#include <Eigen/Dense>
#include <complex>
#include <algorithm>

#include "theia/math/polynomial.h"
#include "theia/sfm/pose/util.h"


PYBIND11_MODULE(pytheia, m) {


  //m.def("PoseFromThreePoints", theia::PoseFromThreePointsWrapper);


}
