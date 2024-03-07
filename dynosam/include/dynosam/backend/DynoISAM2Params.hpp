/*
 *   Copyright (c) 2024 ACFR-RPG, University of Sydney, Jesse Morris (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 */

#pragma once

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/nonlinear/DoglegOptimizerImpl.h>

#include <string>
#include <variant>

namespace dyno {

using namespace gtsam;

/**
 * @ingroup isam2
 * Parameters for ISAM2 using Gauss-Newton optimization.  Either this class or
 * DynoISAM2DoglegParams should be specified as the optimizationParams in
 * DynoISAM2Params, which should in turn be passed to ISAM2(const DynoISAM2Params&).
 */
struct DynoISAM2GaussNewtonParams {
  double
      wildfireThreshold;  ///< Continue updating the linear delta only when
                          ///< changes are above this threshold (default: 0.001)

  /** Specify parameters as constructor arguments */
  DynoISAM2GaussNewtonParams(
      double _wildfireThreshold =
          0.001  ///< see DynoISAM2GaussNewtonParams public variables,
                 ///< DynoISAM2GaussNewtonParams::wildfireThreshold
      )
      : wildfireThreshold(_wildfireThreshold) {}

  void print(const std::string str = "") const {
    using std::cout;
    cout << str << "type:              DynoISAM2GaussNewtonParams\n";
    cout << str << "wildfireThreshold: " << wildfireThreshold << "\n";
    cout.flush();
  }

  double getWildfireThreshold() const { return wildfireThreshold; }
  void setWildfireThreshold(double wildfireThreshold) {
    this->wildfireThreshold = wildfireThreshold;
  }
};

/**
 * @ingroup isam2
 * Parameters for ISAM2 using Dogleg optimization.  Either this class or
 * DynoISAM2GaussNewtonParams should be specified as the optimizationParams in
 * DynoISAM2Params, which should in turn be passed to ISAM2(const DynoISAM2Params&).
 */
struct DynoISAM2DoglegParams {
  double initialDelta;  ///< The initial trust region radius for Dogleg
  double
      wildfireThreshold;  ///< Continue updating the linear delta only when
                          ///< changes are above this threshold (default: 1e-5)
  DoglegOptimizerImpl::TrustRegionAdaptationMode
      adaptationMode;  ///< See description in
                       ///< DoglegOptimizerImpl::TrustRegionAdaptationMode
  bool
      verbose;  ///< Whether Dogleg prints iteration and convergence information

  /** Specify parameters as constructor arguments */
  DynoISAM2DoglegParams(
      double _initialDelta = 1.0,  ///< see DynoISAM2DoglegParams::initialDelta
      double _wildfireThreshold =
          1e-5,  ///< see DynoISAM2DoglegParams::wildfireThreshold
      DoglegOptimizerImpl::TrustRegionAdaptationMode _adaptationMode =
          DoglegOptimizerImpl::
              SEARCH_EACH_ITERATION,  ///< see DynoISAM2DoglegParams::adaptationMode
      bool _verbose = false           ///< see DynoISAM2DoglegParams::verbose
      )
      : initialDelta(_initialDelta),
        wildfireThreshold(_wildfireThreshold),
        adaptationMode(_adaptationMode),
        verbose(_verbose) {}

  void print(const std::string str = "") const {
    using std::cout;
    cout << str << "type:              DynoISAM2DoglegParams\n";
    cout << str << "initialDelta:      " << initialDelta << "\n";
    cout << str << "wildfireThreshold: " << wildfireThreshold << "\n";
    cout << str
         << "adaptationMode:    " << adaptationModeTranslator(adaptationMode)
         << "\n";
    cout.flush();
  }

  double getInitialDelta() const { return initialDelta; }
  double getWildfireThreshold() const { return wildfireThreshold; }
  std::string getAdaptationMode() const {
    return adaptationModeTranslator(adaptationMode);
  }
  bool isVerbose() const { return verbose; }
  void setInitialDelta(double initialDelta) {
    this->initialDelta = initialDelta;
  }
  void setWildfireThreshold(double wildfireThreshold) {
    this->wildfireThreshold = wildfireThreshold;
  }
  void setAdaptationMode(const std::string& adaptationMode) {
    this->adaptationMode = adaptationModeTranslator(adaptationMode);
  }
  void setVerbose(bool verbose) { this->verbose = verbose; }

  std::string adaptationModeTranslator(
      const DoglegOptimizerImpl::TrustRegionAdaptationMode& adaptationMode)
      const;
  DoglegOptimizerImpl::TrustRegionAdaptationMode adaptationModeTranslator(
      const std::string& adaptationMode) const;
};

/**
 * @ingroup isam2
 * Parameters for the ISAM2 algorithm.  Default parameter values are listed
 * below.
 */
typedef FastMap<char, Vector> ISAM2ThresholdMap;
typedef ISAM2ThresholdMap::value_type ISAM2ThresholdMapValue;
struct DynoISAM2Params {
  typedef std::variant<DynoISAM2GaussNewtonParams, DynoISAM2DoglegParams>
      OptimizationParams;  ///< Either DynoISAM2GaussNewtonParams or
                           ///< DynoISAM2DoglegParams
  typedef std::variant<double, FastMap<char, Vector> >
      RelinearizationThreshold;  ///< Either a constant relinearization
                                 ///< threshold or a per-variable-type set of
                                 ///< thresholds

  /** Optimization parameters, this both selects the nonlinear optimization
   * method and specifies its parameters, either DynoISAM2GaussNewtonParams or
   * DynoISAM2DoglegParams.  In the former, Gauss-Newton optimization will be used
   * with the specified parameters, and in the latter Powell's dog-leg
   * algorithm will be used with the specified parameters.
   */
  OptimizationParams optimizationParams;

  /** Only relinearize variables whose linear delta magnitude is greater than
   * this threshold (default: 0.1).  If this is a FastMap<char,Vector> instead
   * of a double, then the threshold is specified for each dimension of each
   * variable type.  This parameter then maps from a character indicating the
   * variable type to a Vector of thresholds for each dimension of that
   * variable.  For example, if Pose keys are of type TypedSymbol<'x',Pose3>,
   * and landmark keys are of type TypedSymbol<'l',Point3>, then appropriate
   * entries would be added with:
   * \code
     FastMap<char,Vector> thresholds;
     thresholds['x'] = (Vector(6) << 0.1, 0.1, 0.1, 0.5, 0.5, 0.5).finished();
   // 0.1 rad rotation threshold, 0.5 m translation threshold thresholds['l'] =
   Vector3(1.0, 1.0, 1.0);                // 1.0 m landmark position threshold
     params.relinearizeThreshold = thresholds;
     \endcode
   */
  RelinearizationThreshold relinearizeThreshold;

  int relinearizeSkip;  ///< Only relinearize any variables every
                        ///< relinearizeSkip calls to ISAM2::update (default:
                        ///< 10)

  bool enableRelinearization;  ///< Controls whether ISAM2 will ever relinearize
                               ///< any variables (default: true)

  bool evaluateNonlinearError;  ///< Whether to evaluate the nonlinear error
                                ///< before and after the update, to return in
                                ///< ISAM2Result from update()

  enum Factorization { CHOLESKY, QR };
  /** Specifies whether to use QR or CHOESKY numerical factorization (default:
   * CHOLESKY). Cholesky is faster but potentially numerically unstable for
   * poorly-conditioned problems, which can occur when uncertainty is very low
   * in some variables (or dimensions of variables) and very high in others.  QR
   * is slower but more numerically stable in poorly-conditioned problems.  We
   * suggest using the default of Cholesky unless gtsam sometimes throws
   * IndefiniteLinearSystemException when your problem's Hessian is actually
   * positive definite.  For positive definite problems, numerical error
   * accumulation can cause the problem to become numerically negative or
   * indefinite as solving proceeds, especially when using Cholesky.
   */
  Factorization factorization;

  /** Whether to cache linear factors (default: true).
   * This can improve performance if linearization is expensive, but can hurt
   * performance if linearization is very cleap due to computation to look up
   * additional keys.
   */
  bool cacheLinearizedFactors;

  KeyFormatter
      keyFormatter;  ///< A KeyFormatter for when keys are printed during
                     ///< debugging (default: DefaultKeyFormatter)

  bool enableDetailedResults;  ///< Whether to compute and return
                               ///< ISAM2Result::detailedResults, this can
                               ///< increase running time (default: false)

  /** Check variables for relinearization in tree-order, stopping the check once
   * a variable does not need to be relinearized (default: false). This can
   * improve speed by only checking a small part of the top of the tree.
   * However, variables below the check cut-off can accumulate significant
   * deltas without triggering relinearization. This is particularly useful in
   * exploration scenarios where real-time performance is desired over
   * correctness. Use with caution.
   */
  bool enablePartialRelinearizationCheck;

  /// When you will be removing many factors, e.g. when using ISAM2 as a
  /// fixed-lag smoother, enable this option to add factors in the first
  /// available factor slots, to avoid accumulating nullptr factor slots, at the
  /// cost of having to search for slots every time a factor is added.
  bool findUnusedFactorSlots;

  /**
   * Specify parameters as constructor arguments
   * See the documentation of member variables above.
   */
  DynoISAM2Params(OptimizationParams _optimizationParams = DynoISAM2GaussNewtonParams(),
              RelinearizationThreshold _relinearizeThreshold = 0.1,
              int _relinearizeSkip = 10, bool _enableRelinearization = true,
              bool _evaluateNonlinearError = false,
              Factorization _factorization = DynoISAM2Params::CHOLESKY,
              bool _cacheLinearizedFactors = true,
              const KeyFormatter& _keyFormatter =
                  DefaultKeyFormatter,  ///< see ISAM2::Params::keyFormatter,
              bool _enableDetailedResults = false)
      : optimizationParams(_optimizationParams),
        relinearizeThreshold(_relinearizeThreshold),
        relinearizeSkip(_relinearizeSkip),
        enableRelinearization(_enableRelinearization),
        evaluateNonlinearError(_evaluateNonlinearError),
        factorization(_factorization),
        cacheLinearizedFactors(_cacheLinearizedFactors),
        keyFormatter(_keyFormatter),
        enableDetailedResults(_enableDetailedResults),
        enablePartialRelinearizationCheck(false),
        findUnusedFactorSlots(false) {}

  /// print iSAM2 parameters
  void print(const std::string& str = "") const {
    using std::cout;
    cout << str << "\n";

    static const std::string kStr("optimizationParams:                ");
    if (std::holds_alternative<DynoISAM2GaussNewtonParams>(optimizationParams)) {
      std::get<DynoISAM2GaussNewtonParams>(optimizationParams).print();
    } else if (std::holds_alternative<DynoISAM2DoglegParams>(optimizationParams)) {
      std::get<DynoISAM2DoglegParams>(optimizationParams).print(kStr);
    } else {
      cout << kStr << "{unknown type}\n";
    }

    cout << "relinearizeThreshold:              ";
    if (std::holds_alternative<double>(relinearizeThreshold)) {
      cout << std::get<double>(relinearizeThreshold) << "\n";
    } else {
      cout << "{mapped}\n";
      for (const ISAM2ThresholdMapValue& value :
           std::get<ISAM2ThresholdMap>(relinearizeThreshold)) {
        cout << "                                   '" << value.first
             << "' -> [" << value.second.transpose() << " ]\n";
      }
    }

    cout << "relinearizeSkip:                   " << relinearizeSkip << "\n";
    cout << "enableRelinearization:             " << enableRelinearization
         << "\n";
    cout << "evaluateNonlinearError:            " << evaluateNonlinearError
         << "\n";
    cout << "factorization:                     "
         << factorizationTranslator(factorization) << "\n";
    cout << "cacheLinearizedFactors:            " << cacheLinearizedFactors
         << "\n";
    cout << "enableDetailedResults:             " << enableDetailedResults
         << "\n";
    cout << "enablePartialRelinearizationCheck: "
         << enablePartialRelinearizationCheck << "\n";
    cout << "findUnusedFactorSlots:             " << findUnusedFactorSlots
         << "\n";
    cout.flush();
  }

  /// @name Getters and Setters for all properties
  /// @{

  OptimizationParams getOptimizationParams() const {
    return this->optimizationParams;
  }
  RelinearizationThreshold getRelinearizeThreshold() const {
    return relinearizeThreshold;
  }
  std::string getFactorization() const {
    return factorizationTranslator(factorization);
  }
  KeyFormatter getKeyFormatter() const { return keyFormatter; }

  void setOptimizationParams(OptimizationParams optimizationParams) {
    this->optimizationParams = optimizationParams;
  }
  void setRelinearizeThreshold(RelinearizationThreshold relinearizeThreshold) {
    this->relinearizeThreshold = relinearizeThreshold;
  }
  void setFactorization(const std::string& factorization) {
    this->factorization = factorizationTranslator(factorization);
  }
  void setKeyFormatter(KeyFormatter keyFormatter) {
    this->keyFormatter = keyFormatter;
  }

  GaussianFactorGraph::Eliminate getEliminationFunction() const {
    return factorization == CHOLESKY
               ? (GaussianFactorGraph::Eliminate)EliminatePreferCholesky
               : (GaussianFactorGraph::Eliminate)EliminateQR;
  }

  /// @}

  /// @name Some utilities
  /// @{

  static Factorization factorizationTranslator(const std::string& str);
  static std::string factorizationTranslator(const Factorization& value);

  /// @}
};

} //dyno
