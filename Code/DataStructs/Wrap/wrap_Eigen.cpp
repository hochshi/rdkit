//
// Created by SH on 05/11/2017.
//

#include <RDBoost/python.h>
#include <RDBoost/Wrap.h>

//#include <RDGeneral/BoostStartInclude.h>
//#include <boost/cstdint.hpp>
//#include <RDGeneral/BoostEndInclude.h>

#include <RDGeneral/types.h>
//#include <RDGeneral/Invariant.h>
#include <RDBoost/PySequenceHolder.h>

#include <numpy/arrayobject.h>
//#include <boost/python/numpy.hpp>
//#include <numpy/npy_common.h>
//#include <RDBoost/import_array.h>
//#include <RDBoost/pyint_api.h>
//#include <Eigen/SparseCore>

namespace python = boost::python;

#include <DataStructs/EigenTypes.h>

using namespace RDKit;

//typedef int64_t WEVIndex;
//typedef eigen::Matrix<double, eigen::Dynamic, 1> VectorXd;
//typedef eigen::SparseMatrix<double, eigen::ColMajor, WEVIndex> SVectorXd;
//typedef eigen::Map<VectorXd> MVectorXd;
//typedef eigen::Map<SVectorXd> MSVectorXd;


template <typename VectorType, typename MapType>
class VectorHelper {
public:
  VectorHelper() {};
  ~VectorHelper() {};


  static MapType * Map(python::object toMap, WEVIndex m) {
    Py_Initialize();
    PyArrayObject *toMapP = (PyArrayObject *)toMap.ptr();
    return new MapType((double *) PyArray_DATA(toMapP), m);
//    static eigen::Map<VectorType> mapVec((double *) PyArray_DATA(toMap), m);
//    return mapVec.derived();
  }

  //sparse mapping
  // Super hack here :(
  static MapType * SparseMap(WEVIndex size, WEVIndex nnz, python::object  outerIndices, python::object  innerIndices, python::object values) {
    Py_Initialize();
    PyArrayObject *outerIndicesP = (PyArrayObject *)outerIndices.ptr();
    PyArrayObject *innerIndicesP = (PyArrayObject *)innerIndices.ptr();
    PyArrayObject *valuesP = (PyArrayObject *)values.ptr();
    return new MapType(size, 1, nnz,
//    static eigen::Map<VectorType> mapVec(size, 1, nnz,
                          (WEVIndex *) PyArray_DATA(outerIndicesP),
                          (WEVIndex *) PyArray_DATA(innerIndicesP),
                          (double *) PyArray_DATA(valuesP));
//    return mapVec.derived();
  }

//  static void resize(VectorType& self, WEVIndex size) { self.resize(size, 0); }
  static double dot(const MapType& self, const MapType& other){ return self.cwiseProduct(other).sum(); }
  static VectorType cwiseMax(const MapType& self, const MapType& other){ return self.cwiseMax(other); }
  static VectorType cwiseMin(const MapType& self, const MapType& other){ return self.cwiseMin(other); }
  static double getVal(MapType& self, WEVIndex index) { return self.coeffRef(index, 0); }
  static void setVal(MapType& self, WEVIndex index, double newVal) { self.coeffRef(index, 0) = newVal; }
  static WEVIndex size(MapType& self) { return self.size();}
  static WEVIndex rows(MapType& self) { return self.rows();}
  static WEVIndex cols(MapType& self) { return self.cols();}
  static double sum(MapType& self) { return self.sum(); }
  static VectorType __neg__(const MapType& a){ return -a; };
  static VectorType __add__(const MapType& a, const MapType& b){ return a+b; }
  static VectorType __sub__(const MapType& a, const MapType& b){ return a-b; }
  static VectorType __iadd__(MapType& a, const MapType& b){ a+=b; return a; };
  static VectorType __isub__(MapType& a, const MapType& b){ a-=b; return a; };
  static VectorType __mulScalar(const MapType& a, const MapType& b){ return a*b; }
  static VectorType __mul(MapType& a, const double b){ return a*b;}

};


using namespace boost::python;
using namespace boost;
template <typename VectorType, typename MapType>
static void wrapOne(const char *className) {

  class_<MapType, shared_ptr<MapType>>(className, "Eigen dense vectorXd", no_init)
//  class_<VectorType>(className, "Eigen dense vectorXd", no_init)
//      .def("resize", &VectorHelper<MVectorType, VectorType>::resize, python::arg("size"))
      .def("__len__", &VectorHelper<VectorType, MapType>::size)
      .def("dot", &VectorHelper<VectorType, MapType>::dot, python::arg("other"), "Dot product with *other*.")
      .def("__setitem__", &VectorHelper<VectorType, MapType>::setVal, python::arg("index"), python::arg("newVal"),
           "Set the value at a specified location")
      .def("__getitem__", &VectorHelper<VectorType, MapType>::getVal, python::arg("index"),
           "Get the value at a specified location")
      .def("sum", &VectorHelper<VectorType, MapType>::sum, "Sum of all elements")
      .def("size", &VectorHelper<VectorType, MapType>::size, "Size")
      .def("rows", &VectorHelper<VectorType, MapType>::rows, "Rows")
      .def("cols", &VectorHelper<VectorType, MapType>::cols, "Cols")
      .def("cwiseMax", &VectorHelper<VectorType, MapType>::cwiseMax, python::arg("other"))
      .def("cwiseMin", &VectorHelper<VectorType, MapType>::cwiseMin, python::arg("other"))
      .def("__neg__", &VectorHelper<VectorType, MapType>::__neg__)
      .def("__add__", &VectorHelper<VectorType, MapType>::__add__)
//      .def("__iadd__", &VectorHelper<VectorType, MapType>::__iadd__)
      .def("__sub__", &VectorHelper<VectorType, MapType>::__sub__)
//      .def("__isub__", &VectorHelper<VectorType, MapType>::__isub__)
      .def("__mul__", &VectorHelper<VectorType, MapType>::__mul)
      .def("__mul__", &VectorHelper<VectorType, MapType>::__mulScalar)
      ;
}

template <typename VectorType, typename MapType>
typename boost::enable_if<std::is_same<MapType, MVectorXd>, VectorType>::type
static wrapMap(const char *className) {

  char helperName[80];

  strcpy(helperName, className);
  strcat(helperName, "Helper");

  class_<VectorHelper<VectorType, MapType>>(helperName)
      .def("Map", &VectorHelper<VectorType, MapType>::Map,
           (python::args("toMap"), python::args("m")),
           python::return_value_policy<python::manage_new_object>())
      .staticmethod("Map")
      ;

//  wrapOne<VectorType>(className);
}

template <typename VectorType, typename MapType>
typename boost::enable_if<std::is_same<MapType, MSVectorXd>, VectorType>::type
static wrapMap(const char *className) {

  char helperName[80];

  strcpy(helperName, className);
  strcat(helperName, "Helper");

  class_<VectorHelper<VectorType, MapType>>(helperName)
      .def("Map", &VectorHelper<VectorType, MapType>::SparseMap,
           (python::args("size"), python::args("nnz"),
               python::args("outerIndices"),
               python::args("innerIndices"),
               python::args("values")),
           python::return_value_policy<python::manage_new_object>(),
           "Must be called with: total size, nnz, column position array ([0,size]) as int64, row position array as in64 and values as double")
      .staticmethod("Map")
      ;
//  wrapOne<VectorType>(className);
}

struct Eigen_wrapper {
  static void wrap() {
    wrapMap<VectorXd, MVectorXd>("MVectorXd");
    wrapMap<SVectorXd, MSVectorXd>("MSVectorXd");
    wrapOne<VectorXd, VectorXd>("VectorXd");
    wrapOne<VectorXd, MVectorXd>("MVectorXd");
    wrapOne<SVectorXd, SVectorXd>("SVectorXd");
    wrapOne<SVectorXd, MSVectorXd>("MSVectorXd");
  }
};

void wrap_eigen(){ Eigen_wrapper::wrap(); }