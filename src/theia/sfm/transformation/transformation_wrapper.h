#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <tuple>

namespace theia {

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double>  AlignPointCloudsUmeyamaWrapper(const std::vector<Eigen::Vector3d>& left,
                             const std::vector<Eigen::Vector3d>& right);

std::tuple<Eigen::Matrix3d, Eigen::Vector3d, double> AlignPointCloudsUmeyamaWithWeightsWrapper(
    const std::vector<Eigen::Vector3d>& left,
    const std::vector<Eigen::Vector3d>& right,
    const std::vector<double>& weights);

std::tuple<std::vector<Eigen::Matrix<double,4,1>>, std::vector<Eigen::Vector3d>, std::vector<double>> GdlsSimilarityTransformWrapper(const std::vector<Eigen::Vector3d>& ray_origin,
                             const std::vector<Eigen::Vector3d>& ray_direction,
                             const std::vector<Eigen::Vector3d>& world_point);



}  // namespace theia
