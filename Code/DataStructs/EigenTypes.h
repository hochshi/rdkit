//
// Created by SH on 12/11/2017.
//

#ifndef RDKIT_EIGENTYPES_H
#define RDKIT_EIGENTYPES_H

#include <Eigen/SparseCore>
namespace eigen = Eigen;

namespace RDKit {
  typedef int64_t WEVIndex;
  typedef eigen::Matrix<double, eigen::Dynamic, 1> VectorXd;
  typedef eigen::SparseMatrix<double, eigen::ColMajor, WEVIndex> SVectorXd;
  typedef eigen::Map<VectorXd> MVectorXd;
  typedef eigen::Map<SVectorXd> MSVectorXd;
}

#endif //RDKIT_EIGENTYPES_H
