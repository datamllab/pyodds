PyODDS
======

**PyODDS** is an end-to end **Python** system for **outlier**
**detection** with **database** **support**. PyODDS provides outlier
detection algorithms which meet the demands for users in different
fields, w/wo data science or machine learning background. PyODDS gives
the ability to execute machine learning algorithms in-database without
moving data out of the database server or over the network. It also
provides access to a wide range of outlier detection algorithms,
including statistical analysis and more recent deep learning based
approaches.

PyODDS is featured for:

-  **Full Stack Service** which supports operations and maintenances
   from light-weight SQL based database to back-end machine learning
   algorithms and makes the throughput speed faster;

-  **State-of-the-art Anomaly Detection Approaches** including
   **Statistical/Machine Learning/Deep Learning** models with unified
   APIs and detailed documentation;

-  **Powerful Data Analysis Mechanism** which supports both **static and
   time-series data** analysis with flexible time-slice(sliding-window)
   segmentation.

API Demo
^^^^^^^^

.. code:: sh

   from utils.import_algorithm import algorithm_selection
   from utils.utilities import output_performance,connect_server,query_data

   # connect to the database
   conn,cursor=connect_server(host, user, password)

   # query data from specific time range
   data = query_data(database_name,table_name,start_time,end_time)

   # train the anomaly detection algorithm
   clf = algorithm_selection(algorithm_name)
   clf.fit(X_train)

   # get outlier result and scores
   prediction_result = clf.predict(X_test)
   outlierness_score = clf.decision_function(test)

   # visualize the prediction_result
   visualize_distribution(X_test,prediction_result,outlierness_score)

Quick Start
^^^^^^^^^^^

.. code:: sh

   python main.py --ground_truth --visualize_distribution

Results are shown as
--------------------

.. code:: sh

   connect to TDengine success
   Load dataset and table
   Loading cost: 0.151061 seconds
   Load data successful
   Start processing:
   100%|***************| 10/10 [00:00<00:00, 14.02it/s]
   ==============================
   Results in Algorithm dagmm are:
   accuracy_score: 0.98
   precision_score: 0.99
   recall_score: 0.99
   f1_score: 0.99
   processing time: 15.330137 seconds
   roc_auc_score: 0.99
   ==============================
   connection is closed


