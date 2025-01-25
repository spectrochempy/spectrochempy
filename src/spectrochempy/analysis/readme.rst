The structure set up for the homogenization of the analysis models is as follows:

1. Each decomposition model derives from the DecompositionAnalysis base class,
   which in turn derives from the AnalysisConfigurable class. For now, the model
   classes EFA, FASTICA (not sure if this is really useful), MCRALS, NNMF, PCA,
   SIMPLISMA, and SVD have been implemented with this new configuration.

2. Each Linear Regression model derives from the base class LinearRegressionAnalysis,
   which also derives from AnalysisConfigurable. For the moment LSTSQ and NNLS have been
   adapted to this new configuration.

3. Classes that do not fit into the decomposition or linear regression models can be
   derived directly from the AnalysisConfigurable class.

4. The AnalysisConfigurable class has as public method: the "fit" method which,
   as in sckit-learn, launches the basic process. This function should not normally be
   redefined in the subclasses (see example PCA and MCRALS) but only the private
   method "_fit" which is specific to each model. The "fit" method always takes at
   least one NDDataset as input or an array-like which is internally transformed
   immediately into NDDataset (X). On the other hand, the private method "_fit"
   which is called internally only works on the data (np.ndarray):
   This allows to keep the performances as good as possible by avoiding the
   interpretation of the metadata at each step of the models calculations.

   The "fit" method can take a second parameter Y as in the MCRALS classes.
   Y is either an NDDataset/array-like or a tuple of NDDatasets/array-like. This
   second parameter is used for example in MCRALS or LSTSQ models.

   We note that unlike sklearn, we manage masked data in "fit". The hidden
   rows or columns are removed before "_fit" is called, and then reintroduced
   when the calculation data is returned later.

   The output of fit is stored internally in the private variable _outfit,
   so that is can be used later by specific public methods.

   The AnalysisConfigurable class also has other public methods or properties:
   "help", "log", "parameters", "reset", or "X" that are available and therefore
   do not need to be rewritten in the derived models.

5. In the derived class DecompositionAnalysis there are public methods designed
   for decomposition models: "transform" (reduce) and "inverse_transform" (reconstruct)
   and the "components" property (with the associated "get_components" method).
   As for "fit", we do not normally touch these methods in the subclasses.
   We simply define the private methods, "_transform" and "_inverse transform",
   which work with ndarray and not with NDDataset. Also, "_get_components"
   is used to output the number of components used. The data to be returned
   to the user is wrapped by the utility decorator _wrap_ndarray_output_to_nddataset.
   In this way, NDDatasets are returned without any intervention other
   than the possible addition of metadata.
