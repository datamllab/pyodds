APIs Cheatsheet
---------------

The Full API Reference can be found in `handbook`_.

-  **connect_server(hostname,username,password)**: Connect to Apache
   backend TDengine Service.

-  **query_data(connection,cursor,database_name,table_name,start_time,end_time)**:
   Query data from table *table_name* in database *database_name* within
   a given time range.

-  **algorithm_selection(algorithm_name,contamination)**: Select an
   algorithm as detector.

-  **fit(X)**: Fit *X* to detector.

-  **predict(X)**: Predict if instance in *X* is outlier or not.

-  **decision_function(X)**: Output the anomaly score of instances in
   *X*.

-  **output_performance(algorithm_name,ground_truth,prediction_result,outlierness_score)**:
   Output the prediction result as evaluation matrix in *Accuracy*,
   *Precision*, *Recall*, *F1 Score*, *ROC-AUC Score*, *Cost time*.

-  **visualize_distribution(X,prediction_result,outlierness_score)**:
   Visualize the detection result with the the data distribution.

-  **visualize_outlierscore(outlierness_score,prediction_result,contamination)**
   Visualize the detection result with the outlier score.

.. _handbook: https://https://pyodds.github.io/
