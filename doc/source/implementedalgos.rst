Implemented Algorithms
----------------------

Statistical Based Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

================ ===================================== =====================================
Methods          Algorithm                             Class API
================ ===================================== =====================================
CBLOF            Clustering-Based Local Outlier Factor :class:``algo.cblof.CBLOF``
HBOS             Histogram-based Outlier Score         :class:``algo.hbos.HBOS``
IFOREST          Isolation Forest                      :class:``algo.iforest.IFOREST``
KNN              k-Nearest Neighbors                   :class:``algo.knn.KNN``
LOF              Local Outlier Factor                  :class:``algo.cblof.CBLOF``
OCSVM            One-Class Support Vector Machines     :class:``algo.ocsvm.OCSVM``
PCA              Principal Component Analysis          :class:``algo.pca.PCA``
RobustCovariance Robust Covariance                     :class:``algo.robustcovariance.RCOV``
SOD              Subspace Outlier Detection            :class:``algo.sod.SOD``
================ ===================================== =====================================

Deep Learning Based Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

+---------------------+-----------------------+-----------------------+
| Methods             | Algorithm             | Class API             |
+=====================+=======================+=======================+
| autoencoder         | Outlier detection     | :class:``algo.autoenc |
|                     | using replicator      | oder.AUTOENCODER``    |
|                     | neural networks       |                       |
+---------------------+-----------------------+-----------------------+
| dagmm               | Deep autoencoding     | :class:``algo.dagmm.D |
|                     | gaussian mixture      | AGMM``                |
|                     | model for             |                       |
|                     | unsupervised anomaly  |                       |
|                     | detection             |                       |
+---------------------+-----------------------+-----------------------+

Time Serie Methods
^^^^^^^^^^^^^^^^^^

+---------------------+-----------------------+-----------------------+
| Methods             | Algorithm             | Class API             |
+=====================+=======================+=======================+
| lstmad              | Long short term       | :class:``algo.lstm_ad |
|                     | memory networks for   | .LSTMAD``             |
|                     | anomaly detection in  |                       |
|                     | time series           |                       |
+---------------------+-----------------------+-----------------------+
| lstmencdec          | LSTM-based            | :class:``algo.lstm_en |
|                     | encoder-decoder for   | c_dec_axl.LSTMED``    |
|                     | multi-sensor anomaly  |                       |
|                     | detection             |                       |
+---------------------+-----------------------+-----------------------+
| luminol             | Linkedin’s luminol    | :class:``algo.luminol |
|                     |                       | .LUMINOL``            |
+---------------------+-----------------------+-----------------------+
