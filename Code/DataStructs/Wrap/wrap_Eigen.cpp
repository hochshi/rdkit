//
// Created by SH on 05/11/2017.
//

#include <RDBoost/python.h>

#include <RDGeneral/types.h>
#include <RDBoost/PySequenceHolder.h>

#include <numpy/arrayobject.h>
#include <boost/python/numpy.hpp>

#include <DataStructs/EigenTypes.h>

using namespace RDKit;

template <typename VectorType>
class VectorHelper {
public:
  VectorHelper() {};
  ~VectorHelper() {};

  static VectorType * Create() {
    return new VectorType();
  }

  static void Map(VectorXd& vec , python::object toMap, WEVIndex size) {
    Py_Initialize();
    PyArrayObject *toMapP = (PyArrayObject *)toMap.ptr();
    double *toMapData = (double *) PyArray_DATA(toMapP);
    vec.resize(size, 1);
    for (WEVIndex i=0; i < size; i++ ) {
      vec.coeffRef(i, 0) = toMapData[i];
    }
  }

  static void SparseMap(SVectorXd& vec, python::object  innerIndices, python::object values, WEVIndex nnz,
                        WEVIndex size) {
    Py_Initialize();
    PyArrayObject *innerIndicesP = (PyArrayObject *)innerIndices.ptr();
    PyArrayObject *valuesP = (PyArrayObject *)values.ptr();
    WEVIndex *rowPos = (WEVIndex *) PyArray_DATA(innerIndicesP);
    double *valuesData = (double *) PyArray_DATA(valuesP);
    vec.resize(size, 1);
    vec.reserve(nnz);
    for (WEVIndex i=0; i<nnz; i++) {
      vec.insert(rowPos[i],0) = valuesData[i];
    }
    vec.makeCompressed();
  }

  static WEVIndex resize(VectorType& self, WEVIndex new_size) {self.resize(new_size, 1); return self.size();}
  static double dot(const VectorType& self, const VectorType& other){ return self.cwiseProduct(other).sum(); }
  static VectorType cwiseMax(const VectorType& self, const VectorType& other){ return self.cwiseMax(other); }
  static VectorType cwiseMin(const VectorType& self, const VectorType& other){ return self.cwiseMin(other); }
  static double getVal(VectorType& self, WEVIndex index) { return self.coeffRef(index, 0); }
  static void setVal(VectorType& self, WEVIndex index, double newVal) { self.coeffRef(index, 0) = newVal; }
  static WEVIndex size(VectorType& self) { return self.size();}
  static WEVIndex rows(VectorType& self) { return self.rows();}
  static WEVIndex cols(VectorType& self) { return self.cols();}
  static double sum(VectorType& self) { return self.sum(); }
  static VectorType __neg__(const VectorType& a){ return -a; };
  static VectorType __add__(const VectorType& a, const VectorType& b){ return a+b; }
  static VectorType __sub__(const VectorType& a, const VectorType& b){ return a-b; }
  static VectorType __iadd__(VectorType& a, const VectorType& b){ a+=b; return a; };
  static VectorType __isub__(VectorType& a, const VectorType& b){ a-=b; return a; };
  static VectorType __mul(const VectorType& a, const VectorType& b){ return a.cwiseProduct(b); }
  static VectorType __mulVec(const VectorType& a, const VectorType& b){ return a.cwiseProduct(b); }
  static VectorType __mulScalar(VectorType& a, const double b){ return a*b;}

};

namespace np = boost::python::numpy;
template <typename VectorType>
class WeightedTanimotoUFunc {
public:
  VectorType wv;
  VectorType ref_fp;
  double v1Sum;

  WeightedTanimotoUFunc(VectorType fp, VectorType weightVec): wv(weightVec), ref_fp(fp){
    v1Sum = 0.0;
    v1Sum = ref_fp.cwiseProduct(wv).sum();
  };

  double getV1Sum() const {
    return v1Sum;
  }

  void calcWeightedVectParams(const VectorType &ev2, double &v2Sum, double &andSum) {
    v2Sum = andSum = 0.0;
    v2Sum = ev2.cwiseProduct(wv).sum();
    andSum = ref_fp.cwiseMin(ev2).cwiseProduct(wv).sum();
  }

  double WeightedTanimotoSimilarity(const VectorType &ev2) {
    return WeightedTanimotoSimilarity1(ev2);
  }

  double WeightedTanimotoSimilarity1(const VectorType &ev2) {
    double v2Sum = 0.0;
    double andSum = 0.0;

    calcWeightedVectParams(ev2, v2Sum, andSum);

    double denom = v1Sum + v2Sum - andSum;
    double sim;

    if (fabs(denom) < 1e-6) {
      sim = 0.0;
    } else {
      sim = andSum / denom;
    }
    // std::cerr<<" "<<v1Sum<<" "<<v2Sum<<" " << numer << " " << sim <<std::endl;
    return sim;
  }

  python::list WeightedTanimotoSimilarity(const python::list &in_arr ) {
    return WeightedTanimotoSimilarity2(in_arr);
  }

  python::list WeightedTanimotoSimilarity2(const python::list &in_arr ) {
    python::list res;
    unsigned int nsev = python::extract<unsigned int>(in_arr.attr("__len__")());
    for (unsigned int i = 0; i < nsev; ++i) {
      double simVal;
      const VectorType &ev2 = python::extract<VectorType>(in_arr[i])();
      simVal = WeightedTanimotoSimilarity1(ev2);
      res.append(simVal);
    }
    return res;
//    np::dtype out_dtype = np::dtype::get_builtin<double>();
//    np::ndarray out_array = np::zeros(in_arr.get_nd(), in_arr.get_shape(), out_dtype);
//    np::multi_iter iter = make_multi_iter(in_arr, out_array);
//    while (iter.not_done())
//    {
////      VectorType * argument = reinterpret_cast<VectorType*>(iter.get_data(0));
//      VectorType * argument = reinterpret_cast<python::object*>(iter.get_data(0));
//      double * result = reinterpret_cast<double *>(iter.get_data(1));
//      *result = WeightedTanimotoSimilarity1(*argument);
//      iter.next();
//    }
//    return out_array.scalarize();
  }

};

template <typename VectorType>
static void wrapOne(const char *className) {

  python::class_<VectorType, boost::shared_ptr<VectorType> >(className, "Eigen sparse vectorXd", python::no_init)
      .def("resize", &VectorHelper<VectorType>::resize, python::arg("size"))
      .def("__len__", &VectorHelper<VectorType>::size)
      .def("dot", &VectorHelper<VectorType>::dot, python::arg("other"), "Dot product with *other*.")
      .def("__setitem__", &VectorHelper<VectorType>::setVal, python::arg("index"), python::arg("newVal"),
           "Set the value at a specified location")
      .def("__getitem__", &VectorHelper<VectorType>::getVal, python::arg("index"),
           "Get the value at a specified location")
      .def("sum", &VectorHelper<VectorType>::sum, "Sum of all elements")
      .def("size", &VectorHelper<VectorType>::size, "Size")
      .def("rows", &VectorHelper<VectorType>::rows, "Rows")
      .def("cols", &VectorHelper<VectorType>::cols, "Cols")
      .def("cwiseMax", &VectorHelper<VectorType>::cwiseMax, python::arg("other"))
      .def("cwiseMin", &VectorHelper<VectorType>::cwiseMin, python::arg("other"))
      .def("__neg__", &VectorHelper<VectorType>::__neg__)
      .def("__add__", &VectorHelper<VectorType>::__add__)
//      .def("__iadd__", &VectorHelper<VectorType, MapType>::__iadd__)
      .def("__sub__", &VectorHelper<VectorType>::__sub__)
//      .def("__isub__", &VectorHelper<VectorType, MapType>::__isub__)
      .def("__mul__", &VectorHelper<VectorType>::__mul)
      .def("__mul__", &VectorHelper<VectorType>::__mulScalar)
      .def("__mul__", &VectorHelper<VectorType>::__mulVec)
      ;
}


template <typename VectorType>
static void wrapWeightedTanimotoUFunc(const char *className) {
  Py_Initialize();
  np::initialize();
  python::class_<WeightedTanimotoUFunc<VectorType>,
      boost::shared_ptr<WeightedTanimotoUFunc<VectorType> > >
      (className,"WeightedTanimotoUFunc", python::init<VectorType, VectorType>())
      .def("__call__", &WeightedTanimotoUFunc<VectorType>::WeightedTanimotoSimilarity1, (python::args("svec")))
      .def("__call__", &WeightedTanimotoUFunc<VectorType>::WeightedTanimotoSimilarity2, (python::args("ndarray")))
      .def("getV1Sum", &WeightedTanimotoUFunc<VectorType>::getV1Sum)
      ;
}

template <typename VectorType>
static void wrapMap(const char *className){};

template <>
void wrapMap<VectorXd>(const char *className) {

  char helperName[80];

  strcpy(helperName, className);
  strcat(helperName, "Helper");

  python::class_<VectorHelper<VectorXd> >(helperName)
      .def("Map", &VectorHelper<VectorXd>::Map,
           (python::args("vec"), python::args("toMap"), python::args("size")))
      .staticmethod("Map")
      .def("Create", &VectorHelper<VectorXd>::Create,
           python::return_value_policy<python::manage_new_object>())
      .staticmethod("Create")
      ;
}

template <>
void wrapMap<SVectorXd>(const char *className) {

  char helperName[80];

  strcpy(helperName, className);
  strcat(helperName, "Helper");

  python::class_<VectorHelper<SVectorXd> >(helperName)
      .def("Map", &VectorHelper<SVectorXd>::SparseMap,
           (python::args("vec"), python::args("innerIndices"),
               python::args("values"),
               python::args("nnz"),
               python::args("size")))
      .staticmethod("Map")
      .def("Create", &VectorHelper<SVectorXd>::Create,
           python::return_value_policy<python::manage_new_object>())
      .staticmethod("Create")
      ;
}

struct Eigen_wrapper {
  static void wrap() {
    wrapMap<VectorXd>("MVectorXd");
    wrapMap<SVectorXd>("MSVectorXd");
    wrapOne<VectorXd>("VectorXd");
//    wrapOne<VectorXd, MVectorXd>("MVectorXd");
    wrapOne<SVectorXd>("SVectorXd");
//    wrapOne<SVectorXd, MSVectorXd>("MSVectorXd");
    wrapWeightedTanimotoUFunc<VectorXd>("WTCVectorXd");
    wrapWeightedTanimotoUFunc<SVectorXd>("WTCSVectorXd");
  }
};

void wrap_eigen(){ Eigen_wrapper::wrap(); }
